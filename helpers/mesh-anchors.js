// backend/helpers/mesh-anchors.js
// Utilities for working with MediaPipe Face Mesh (468 or 478 points).
// Points are expected as an array of { x, y, z } where x/y are normalized [0..1].
// If you've already converted to pixels, these helpers still work the same.

// ---- Common landmark indices (MediaPipe Face Mesh) ----
// These indices are community-standard anchors used in examples and papers.
// They’re stable enough for ROI building & symmetry midline.
const IDX = {
  // Eyes (outer/inner corners)
  LEFT_EYE_OUTER: 33,
  LEFT_EYE_INNER: 133,
  RIGHT_EYE_INNER: 362,
  RIGHT_EYE_OUTER: 263,

  // Optional iris (requires refineLandmarks: true → total points = 478)
  LEFT_IRIS_CENTER: 468,  // approximate center
  RIGHT_IRIS_CENTER: 473, // approximate center

  // Nose
  NOSE_TIP: 1,
  NOSE_BRIDGE: 0,

  // Mouth
  MOUTH_LEFT: 61,
  MOUTH_RIGHT: 291,
  MOUTH_TOP: 13,
  MOUTH_BOTTOM: 14,

  // Chin / jaw
  CHIN: 152,

  // Brows (useful for forehead/glabellar ROIs)
  BROW_LEFT_INNER: 55,
  BROW_LEFT_OUTER: 52,
  BROW_RIGHT_INNER: 285,
  BROW_RIGHT_OUTER: 283,
};

// Safe accessor
function pick(pts, i) {
  const p = pts && pts[i];
  return p ? { x: p.x, y: p.y, z: p.z ?? 0 } : { x: 0, y: 0, z: 0 };
}

// Average of two points
function mid(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 };
}

// Euclidean distance (2D)
function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/**
 * Convert normalized points [0..1] → pixel coordinates.
 * Pass canvas width/height (or image width/height).
 */
function toPixels(pts, width, height) {
  return (pts || []).map(p => ({
    x: p.x * width,
    y: p.y * height,
    z: p.z, // keep z as-is (relative)
  }));
}

/**
 * Compute basic anchors from a 468/478-point mesh.
 * Returns convenient semantic points and centers used by ROI logic.
 */
function anchorsFrom468(pts) {
  const leftEyeOuter  = pick(pts, IDX.LEFT_EYE_OUTER);
  const leftEyeInner  = pick(pts, IDX.LEFT_EYE_INNER);
  const rightEyeInner = pick(pts, IDX.RIGHT_EYE_INNER);
  const rightEyeOuter = pick(pts, IDX.RIGHT_EYE_OUTER);

  const leftEyeCenter  = mid(leftEyeOuter, leftEyeInner);
  const rightEyeCenter = mid(rightEyeOuter, rightEyeInner);
  const eyesCenter     = mid(leftEyeCenter, rightEyeCenter);

  const irisLeft  = pick(pts, IDX.LEFT_IRIS_CENTER);
  const irisRight = pick(pts, IDX.RIGHT_IRIS_CENTER);

  const noseTip    = pick(pts, IDX.NOSE_TIP);
  const noseBridge = pick(pts, IDX.NOSE_BRIDGE);

  const mouthLeft   = pick(pts, IDX.MOUTH_LEFT);
  const mouthRight  = pick(pts, IDX.MOUTH_RIGHT);
  const mouthTop    = pick(pts, IDX.MOUTH_TOP);
  const mouthBottom = pick(pts, IDX.MOUTH_BOTTOM);
  const mouthCenter = mid(mouthLeft, mouthRight);

  const chin = pick(pts, IDX.CHIN);

  const browLeftInner  = pick(pts, IDX.BROW_LEFT_INNER);
  const browLeftOuter  = pick(pts, IDX.BROW_LEFT_OUTER);
  const browRightInner = pick(pts, IDX.BROW_RIGHT_INNER);
  const browRightOuter = pick(pts, IDX.BROW_RIGHT_OUTER);

  return {
    // Eyes
    leftEyeOuter, leftEyeInner, rightEyeOuter, rightEyeInner,
    leftEyeCenter, rightEyeCenter, eyesCenter,
    irisLeft, irisRight,

    // Nose
    noseTip, noseBridge,

    // Mouth
    mouthLeft, mouthRight, mouthTop, mouthBottom, mouthCenter,

    // Chin / jaw
    chin,

    // Brows (forehead / glabella helpers)
    browLeftInner, browLeftOuter, browRightInner, browRightOuter,
  };
}

/**
 * Eye centers (sugar helper).
 */
function eyeCentersFrom468(pts) {
  const a = anchorsFrom468(pts);
  return { left: a.leftEyeCenter, right: a.rightEyeCenter };
}

/**
 * Compute a symmetry midline from the eyes.
 * Returns:
 *  - origin: midpoint between eye centers
 *  - dir: unit direction vector along the eye axis (left→right)
 *  - normal: unit normal vector (perpendicular), convenient for mirroring
 */
function midlineFrom468(pts) {
  const { left, right } = eyeCentersFrom468(pts);
  const dx = right.x - left.x;
  const dy = right.y - left.y;
  const len = Math.hypot(dx, dy) || 1;
  const dir = { x: dx / len, y: dy / len };
  const normal = { x: -dir.y, y: dir.x }; // rotate 90°
  const origin = { x: (left.x + right.x) / 2, y: (left.y + right.y) / 2 };
  return { origin, dir, normal };
}

/**
 * Mirror a point across a midline (returned by midlineFrom468).
 * Useful for RMS symmetry calculations.
 */
function mirrorPointAcrossMidline(p, midline) {
  const { origin, normal } = midline;
  // Vector from origin to point
  const vx = p.x - origin.x;
  const vy = p.y - origin.y;
  // Signed distance along the normal
  const d = vx * normal.x + vy * normal.y;
  // Reflection: subtract 2 * projection on the normal
  return {
    x: p.x - 2 * d * normal.x,
    y: p.y - 2 * d * normal.y,
    z: p.z,
  };
}

/**
 * Compute RMS symmetry in pixels across ALL provided landmarks.
 * If you want IPD-normalized mm, divide by IPD (in px) and multiply by your mm/IPD constant.
 */
function symmetryRmsFrom468(pts) {
  if (!pts || pts.length < 2) return { rmsPx: 0, ipdPx: 0 };
  const { left, right } = eyeCentersFrom468(pts);
  const ipdPx = dist(left, right) || 1;

  const mid = midlineFrom468(pts);
  let sumSq = 0;
  let n = 0;
  for (const p of pts) {
    const pm = mirrorPointAcrossMidline(p, mid);
    const dpx = Math.hypot(p.x - pm.x, p.y - pm.y);
    sumSq += dpx * dpx;
    n++;
  }
  const rmsPx = Math.sqrt(sumSq / Math.max(1, n));
  return { rmsPx, ipdPx };
}

module.exports = {
  IDX,
  anchorsFrom468,
  eyeCentersFrom468,
  midlineFrom468,
  mirrorPointAcrossMidline,
  symmetryRmsFrom468,
  toPixels,
};
