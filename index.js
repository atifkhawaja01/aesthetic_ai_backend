// ============================================================================
// 0) MUST BE FIRST: pretend we're in a browser + patch TextEncoder/Decoder
// ============================================================================
(() => {
  if (typeof global.window === 'undefined') {
    global.window = { document: {} };
    global.self = global.window;
  }
  if (typeof global.navigator === 'undefined') {
    global.navigator = { userAgent: 'node' };
  }
  try {
    const util = require('util');
    const utilNode = require('node:util');
    const Enc = global.TextEncoder || util.TextEncoder || utilNode.TextEncoder;
    const Dec = global.TextDecoder || util.TextDecoder || utilNode.TextDecoder;
    if (typeof global.TextEncoder !== 'function' && Enc) global.TextEncoder = Enc;
    if (typeof global.TextDecoder !== 'function' && Dec) global.TextDecoder = Dec;
    if (typeof util.TextEncoder !== 'function' && global.TextEncoder) util.TextEncoder = global.TextEncoder;
    if (typeof util.TextDecoder !== 'function' && global.TextDecoder) util.TextDecoder = global.TextDecoder;
    if (typeof utilNode.TextEncoder !== 'function' && global.TextEncoder) utilNode.TextEncoder = global.TextEncoder;
    if (typeof utilNode.TextDecoder !== 'function' && global.TextDecoder) utilNode.TextDecoder = global.TextDecoder;
  } catch {}
})();

// ============================================================================
// 1) Deps & app scaffolding
// ============================================================================
const dotenv = require('dotenv');
dotenv.config();
const os = require('os');
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const canvasLib = require('@napi-rs/canvas');
const { createCanvas, Image, ImageData, loadImage } = canvasLib;

// ---- Optional: fast server-side image transcode/resize
let sharp = null;
try { sharp = require('sharp'); } catch { /* optional */ }

// ---- TensorFlow.js (CPU backend).
const tf = require('@tensorflow/tfjs-core');
require('@tensorflow/tfjs-backend-cpu');

process.on('unhandledRejection', (e) => console.error('[unhandledRejection]', e?.stack || e));
process.on('uncaughtException', (e) => console.error('[uncaughtException]', e?.stack || e));

function suppressTfjsSpam(fn) {
  const origWarn = console.warn;
  console.warn = (msg, ...rest) => {
    const s = String(msg || '');
    if (s.includes('already registered') || s.includes('Overwriting the platform with browser.')) return;
    origWarn(msg, ...rest);
  };
  return Promise.resolve().then(fn).finally(() => { console.warn = origWarn; });
}

(async () => {
  await tf.setBackend('cpu');
  await tf.ready();
  try { tf.removeBackend('webgl'); } catch {}
  console.log('TF backend =>', tf.getBackend());
})();

// ============================================================================
// 2) App config
// ============================================================================
const app = express();
app.disable('x-powered-by');  
const allowed = [
  'https://aiaestheticapp.netlify.app', // 👈 replace with your actual Netlify domain
  'http://localhost:5173'          // optional for local testing
];

app.use(cors({
  origin: allowed,
  credentials: true
}));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

const PORT = process.env.PORT || 4000;
const ROOT = __dirname;
const MODELS_DIR = path.join(ROOT, 'models');
const UPLOAD_DIR = path.join(ROOT, 'uploads');
const DATA_DIR = path.join(ROOT, 'data');
const ANALYSES_DB = path.join(DATA_DIR, 'analyses.json');
const TREATMENTS_DB = path.join(DATA_DIR, 'treatments.json');
const USERS_DB = path.join(DATA_DIR, 'users.json');

// ---- NEW: Feature flag for 468 landmarks
const FACE_MESH_ENABLE = String(process.env.FACE_MESH_ENABLE ?? 'true').toLowerCase() !== 'false';

for (const d of [UPLOAD_DIR, DATA_DIR, MODELS_DIR]) {
  if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true });
}
if (!fs.existsSync(ANALYSES_DB)) fs.writeFileSync(ANALYSES_DB, '[]');
if (!fs.existsSync(USERS_DB)) fs.writeFileSync(USERS_DB, '[]');

app.use('/uploads', express.static(UPLOAD_DIR));

const readJSON = (p, fallback) => {
  try { return JSON.parse(fs.readFileSync(p, 'utf8')); } catch { return fallback; }
};
const writeJSON = (p, obj) => fs.writeFileSync(p, JSON.stringify(obj, null, 2));

// ---- Simple in-memory store for uploaded image buffers (auto-expires)
const MEM_TTL_MS = 10 * 60 * 1000; // 10 minutes
const memStore = new Map(); // key -> { buffer, mime, createdAt }

function putMem(key, buffer, mime) {
  memStore.set(key, { buffer, mime, createdAt: Date.now() });
  // auto-expire after TTL
  setTimeout(() => memStore.delete(key), MEM_TTL_MS).unref?.();
}

function getMem(key) {
  const entry = memStore.get(key);
  return entry?.buffer || null;
}

function delMem(key) {
  memStore.delete(key);
}


// ============================================================================
// 3) Uploads (front / left / right) — hardened for Expo, tunnel-friendly
// ============================================================================
// Save everything as .jpg; we’ll transcode to real JPEG with sharp if present.
// Save nothing to disk: keep images entirely in memory
const storage = multer.memoryStorage();

const upload = multer({
  storage,
  limits: { fileSize: 20 * 1024 * 1024, files: 3 }, // 20MB/file, 3 files
  fileFilter: (_req, file, cb) => {
    const type = String(file?.mimetype || '').toLowerCase();
    if (!type.startsWith('image/')) {
      return cb(new multer.MulterError('LIMIT_UNEXPECTED_FILE', 'ONLY_IMAGES_ALLOWED'));
    }
    cb(null, true);
  },
});


// Transcode anything (HEIC/WEBP/PNG/large JPEG) → oriented JPEG ≤1600px (if sharp is available)
async function ensureJpeg(pathOnDisk) {
  if (!sharp) return;
  try {
    await sharp(pathOnDisk)
      .rotate()
      .resize({ width: 1600, height: 1600, fit: 'inside', withoutEnlargement: true })
      .jpeg({ quality: 85, mozjpeg: true })
      .toFile(pathOnDisk + '.tmp.jpg');
    await fs.promises.rename(pathOnDisk + '.tmp.jpg', pathOnDisk);
  } catch (e) {
    console.error('[uploads] sharp transcode failed:', e?.message || e);
  }
}

// Normalize arbitrary input buffer -> properly oriented, resized JPEG (if sharp is present)
async function ensureJpegBuffer(inputBuffer) {
  if (!sharp) return inputBuffer;
  try {
    const out = await sharp(inputBuffer)
      .rotate()
      .resize({ width: 1600, height: 1600, fit: 'inside', withoutEnlargement: true })
      .jpeg({ quality: 85, mozjpeg: true })
      .toBuffer();
    return out;
  } catch (e) {
    console.error('[uploads] sharp in-memory transcode failed:', e?.message || e);
    return inputBuffer; // fall back to original buffer
  }
}


// ============================================================================
// 4) face-api.js importer + canvas monkey-patch
// ============================================================================
let FACEAPI_PATCHED = false;
let FACEAPI_INSTANCE = null;
let MODEL_SOURCE = 'disk'; // 'disk' | 'url'
let MODEL_URL_USED = '';
let LAST_MODEL_ERROR = '';

const DEFAULT_MODELS_URL = (process.env.FACEAPI_MODELS_URL && process.env.FACEAPI_MODELS_URL.trim()) || 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/';
const USE_TINY = String(process.env.FACEAPI_USE_TINY ?? 'true').toLowerCase() !== 'false';

const expectedModelFiles = [
  'tiny_face_detector_model-weights_manifest.json',
  'face_landmark_68_tiny_model-weights_manifest.json',
  'age_gender_model-weights_manifest.json',
  'face_expression_model-weights_manifest.json',
  'ssd_mobilenetv1_model-weights_manifest.json',
  'face_landmark_68_model-weights_manifest.json',
];

function normalizeFaceApi(mod) {
  const cands = [mod, mod?.default, mod?.faceapi, mod?.default?.faceapi, mod?.default?.default];
  for (const m of cands) {
    if (m && m.nets && (m.nets.ssdMobilenetv1 || m.nets.tinyFaceDetector)) return m;
  }
  return null;
}

async function tryImport(spec) {
  try {
    const m = require(spec);
    return { mod: m, via: `require(${spec})` };
  } catch (e1) {
    try {
      const m = await import(spec);
      return { mod: m, via: `import(${spec})` };
    } catch (e2) {
      return { mod: null, via: null, err: `${e1?.message || e1} | ${e2?.message || e2}` };
    }
  }
}

async function importFaceApi() {
  if (FACEAPI_INSTANCE) return FACEAPI_INSTANCE;
  
  const specs = [
    '@vladmandic/face-api/dist/face-api.node.js',
    '@vladmandic/face-api',
    '@vladmandic/face-api/dist/face-api.js',
    '@vladmandic/face-api/dist/face-api.esm.js',
  ];
  
  const tries = [];
  for (const s of specs) {
    const r = await tryImport(s);
    if (r.mod) {
      const faceapi = normalizeFaceApi(r.mod);
      if (faceapi) {
        console.log(`[face-api] loaded via ${r.via}`);
        FACEAPI_INSTANCE = faceapi;
        return faceapi;
      } else {
        tries.push(`${s} resolved but did not expose nets`);
      }
    } else {
      tries.push(`${s} failed: ${r.err}`);
    }
  }
  
  console.error('[face-api] import attempts:\n- ' + tries.join('\n- '));
  throw new Error('FACEAPI_IMPORT_FAILED');
}

async function getFaceApi() {
  const faceapi = await suppressTfjsSpam(importFaceApi);
  try { tf.removeBackend('webgl'); } catch {}
  if (tf.getBackend() !== 'cpu') { await tf.setBackend('cpu'); }
  
  if (!FACEAPI_PATCHED) {
    const CanvasCtor = createCanvas(1, 1).constructor;
    global.Canvas = CanvasCtor;
    global.HTMLCanvasElement = CanvasCtor;
    global.Image = Image;
    global.HTMLImageElement = Image;
    global.ImageData = ImageData;
    
    const patch = (faceapi.env?.monkeyPatch || faceapi.monkeyPatch);
    if (typeof patch === 'function') {
      patch({ Canvas: CanvasCtor, Image, ImageData });
    } else {
      faceapi.Canvas = CanvasCtor;
      faceapi.Image = Image;
      faceapi.ImageData = ImageData;
      faceapi.env = faceapi.env || {};
      faceapi.env.Canvas = CanvasCtor;
      faceapi.env.Image = Image;
      faceapi.env.ImageData = ImageData;
    }
    FACEAPI_PATCHED = true;
  }
  
  return faceapi;
}

let MODELS_READY = false;
async function loadModelsOnce() {
  if (MODELS_READY) return;
  const faceapi = await getFaceApi();
  const anyExist = expectedModelFiles.some(f => fs.existsSync(path.join(MODELS_DIR, f)));
  
  try {
    const forceUrl = String(process.env.FACEAPI_FORCE_URL || '').toLowerCase() === 'true';
    if (!forceUrl && anyExist && typeof faceapi.nets.ageGenderNet.loadFromDisk === 'function') {
      if (USE_TINY && fs.existsSync(path.join(MODELS_DIR, 'tiny_face_detector_model-weights_manifest.json'))) {
        await faceapi.nets.tinyFaceDetector.loadFromDisk(MODELS_DIR);
        if (fs.existsSync(path.join(MODELS_DIR, 'face_landmark_68_tiny_model-weights_manifest.json'))) {
          await faceapi.nets.faceLandmark68TinyNet.loadFromDisk(MODELS_DIR);
        } else {
          await faceapi.nets.faceLandmark68Net.loadFromDisk(MODELS_DIR);
        }
      } else {
        await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODELS_DIR);
        await faceapi.nets.faceLandmark68Net.loadFromDisk(MODELS_DIR);
      }
      // ensure these are loaded for fallbacks
      await faceapi.nets.ageGenderNet.loadFromDisk(MODELS_DIR);
      await faceapi.nets.faceExpressionNet.loadFromDisk(MODELS_DIR);
      MODEL_SOURCE = 'disk';
      MODEL_URL_USED = '';
      console.log('FaceAPI models loaded from', MODELS_DIR, '| tiny =', USE_TINY);
    } else {
      const url = (DEFAULT_MODELS_URL.endsWith('/') ? DEFAULT_MODELS_URL : DEFAULT_MODELS_URL + '/');
      if (USE_TINY) {
        await faceapi.nets.tinyFaceDetector.loadFromUri(url);
        await faceapi.nets.faceLandmark68TinyNet.loadFromUri(url);
      } else {
        await faceapi.nets.ssdMobilenetv1.loadFromUri(url);
        await faceapi.nets.faceLandmark68Net.loadFromUri(url);
      }
      await faceapi.nets.ageGenderNet.loadFromUri(url);
      await faceapi.nets.faceExpressionNet.loadFromUri(url);
      MODEL_SOURCE = 'url';
      MODEL_URL_USED = url;
      console.log('FaceAPI models loaded from URL:', url, '| tiny =', USE_TINY);
    }
    MODELS_READY = true;
    LAST_MODEL_ERROR = '';
  } catch (e) {
    console.error('Model load failed:', e);
    MODELS_READY = false;
    LAST_MODEL_ERROR = String(e?.message || e);
    throw e;
  }
}

// ============================================================================
// 4b) NEW — Optional FaceMesh 468 loader (TFJS runtime, CPU/Node friendly)
// ============================================================================
let MESH_READY = false;
let MESH_MODEL = null;
let MESH_MODEL_NAME = null;

async function importFaceMeshModel() {
  // Lazy import; works when @tensorflow-models/face-landmarks-detection is installed
  const r = await tryImport('@tensorflow-models/face-landmarks-detection');
  if (!r.mod) throw new Error('FACEMESH_IMPORT_FAILED');
  // normalize namespace
  const mod = r.mod.default || r.mod;
  return mod;
}

async function loadFaceMeshOnce() {
  if (!FACE_MESH_ENABLE) return false;
  if (MESH_READY && MESH_MODEL) return true;
  
  try {
    const fl = await importFaceMeshModel();
    const { SupportedModels, createDetector } = fl;
    // Use TFJS runtime; refineLandmarks=true adds iris points (total 478); we can keep them.
    MESH_MODEL = await createDetector(SupportedModels.MediaPipeFaceMesh, {
      runtime: 'tfjs',
      refineLandmarks: true,
      maxFaces: 1,
    });
    MESH_MODEL_NAME = 'MediaPipeFaceMesh/tfjs';
    MESH_READY = true;
    console.log('[facemesh] ready:', MESH_MODEL_NAME);
    return true;
  } catch (e) {
    console.warn('[facemesh] disabled (load failed):', e?.message || e);
    MESH_READY = false;
    MESH_MODEL = null;
    return false;
  }
}

async function estimateMeshOnCanvas(canvas) {
  if (!MESH_READY || !MESH_MODEL || !canvas) return null;
  try {
    const res = await MESH_MODEL.estimateFaces(canvas, { flipHorizontal: false });
    if (!Array.isArray(res) || res.length === 0) return null;
    const face = res[0];
    // face.keypoints: [{x,y,z,name?}, ...] (478 if refineLandmarks true, else 468)
    const pts = (face.keypoints || []).map(p => ({ x: p.x, y: p.y, z: Number.isFinite(p.z) ? p.z : 0 }));
    if (!pts.length) return null;
    return { points: pts, count: pts.length };
  } catch (e) {
    console.warn('[facemesh] estimate failed:', e?.message || e);
    return null;
  }
}

// Optional: triangulation/topology access (safe require)
let MESH_ANCHORS = null;
try {
  MESH_ANCHORS = require(path.join(__dirname, 'backend', 'helpers', 'mesh-anchors.js'));
  // This is optional and only used downstream for client overlays, if any.
} catch { /* optional */ }

// ============================================================================
// 5) Image helpers + preprocessing & metrics
// ============================================================================
const MAX_DIM = Number(process.env.ANALYSIS_MAX_DIM || 900);
const ASSUMED_IPD_MM = 63; // average IPD in mm

async function loadImageAsCanvas(filePath) {
  const im = await loadImage(filePath);
  const scale = Math.min(1, MAX_DIM / Math.max(im.width, im.height));
  const W = Math.max(1, Math.round(im.width * scale));
  const H = Math.max(1, Math.round(im.height * scale));
  const c = createCanvas(W, H);
  const ctx = c.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(im, 0, 0, W, H);
  return c;
}

// Accepts either a Buffer (preferred) or a file path (string)
async function loadImageAsCanvasFromAny(input) {
  let im;
  if (Buffer.isBuffer(input)) {
    im = await loadImage(input);
  } else {
    im = await loadImage(String(input));
  }
  const scale = Math.min(1, MAX_DIM / Math.max(im.width, im.height));
  const W = Math.max(1, Math.round(im.width * scale));
  const H = Math.max(1, Math.round(im.height * scale));
  const c = createCanvas(W, H);
  const ctx = c.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(im, 0, 0, W, H);
  return c;
}



const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const round = (v) => Math.round(v || 0);
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

// Landmarks → points array across face-api versions
function pointsFromLandmarks(landmarks) {
  if (!landmarks) return [];
  if (typeof landmarks.getPositions === 'function') return landmarks.getPositions();
  if (Array.isArray(landmarks.positions)) return landmarks.positions;
  if (Array.isArray(landmarks)) return landmarks;
  return [];
}

// Get a padded face box for cropped operations (kept for possible future use)
function faceBoxFromDet(canvas, det, landmarks) {
  let x, y, w, h;
  if (det?.detection?.box) {
    const b = det.detection.box;
    x = b.x; y = b.y; w = b.width; h = b.height;
  } else if (landmarks) {
    const pts = pointsFromLandmarks(landmarks);
    const xs = pts.map(p=>p.x), ys = pts.map(p=>p.y);
    x = Math.min(...xs); y = Math.min(...ys);
    const x2 = Math.max(...xs), y2 = Math.max(...ys);
    w = x2 - x; h = y2 - y;
  } else {
    x = 0; y = 0; w = canvas.width; h = canvas.height;
  }
  
  const padX = w * 0.2, padY = h * 0.2;
  const rect = {
    x: Math.round(x - padX/2),
    y: Math.round(y - padY/2),
    w: Math.round(w + padX),
    h: Math.round(h + padY),
  };
  rect.x = clamp(rect.x, 0, canvas.width - 1);
  rect.y = clamp(rect.y, 0, canvas.height - 1);
  rect.w = clamp(rect.w, 20, canvas.width - rect.x);
  rect.h = clamp(rect.h, 20, canvas.height - rect.y);
  return rect;
}

// -------- Basic pixel ops
function luminanceAt(canvas, x, y) {
  const d = canvas.getContext('2d').getImageData(x, y, 1, 1).data;
  return 0.299 * d[0] + 0.587 * d[1] + 0.114 * d[2];
}

function roiClampRect(canvas, x, y, w, h) {
  const X = clamp(Math.round(x), 0, canvas.width - 1);
  const Y = clamp(Math.round(y), 0, canvas.height - 1);
  const W = clamp(Math.round(w), 10, canvas.width - X);
  const H = clamp(Math.round(h), 10, canvas.height - Y);
  return { x: X, y: Y, w: W, h: H };
}

function getROI(ctx, rect) {
  return ctx.getImageData(rect.x, rect.y, rect.w, rect.h);
}

function edgeDensity(imageData, thresh=50) {
  const W=imageData.width, H=imageData.height, data=imageData.data;
  const gray = (i)=>0.299*data[i]+0.587*data[i+1]+0.114*data[i+2];
  let edges=0,total=0;
  for(let y=1;y<H-1;y++){
    for(let x=1;x<W-1;x++){
      const i=(y*W+x)*4;
      const gx=Math.abs(gray(i+4)-gray(i-4));
      const gy=Math.abs(gray(i+4*W)-gray(i-4*W));
      if (gx+gy>thresh) edges++;
      total++;
    }
  }
  return total?edges/total:0;
}

function meanStdRGB(imageData) {
  const d=imageData.data;
  let rSum=0,gSum=0,bSum=0,n=0;
  for (let i=0;i<d.length;i+=4){
    rSum+=d[i]; gSum+=d[i+1]; bSum+=d[i+2]; n++;
  }
  const r=rSum/n,g=gSum/n,b=bSum/n;
  let vr=0,vg=0,vb=0;
  for (let i=0;i<d.length;i+=4){
    vr+=(d[i]-r)**2; vg+=(d[i+1]-g)**2; vb+=(d[i+2]-b)**2;
  }
  vr/=n; vg/=n; vb/=n;
  const lMean = 0.299*r + 0.587*g + 0.114*b;
  const lVar = 0.299*vr + 0.587*vg + 0.114*vb;
  const lStd = Math.sqrt(Math.max(0,lVar));
  return { r,g,b, lMean, lStd };
}

// -------- Simple LAB conversion
function rgbToLabAvg(imageData){
  const { r, g, b } = meanStdRGB(imageData);
  const srgb2lin = (u)=> {
    u/=255;
    return (u<=0.04045)?u/12.92:Math.pow((u+0.055)/1.055,2.4);
  };
  const R=srgb2lin(r), G=srgb2lin(g), B=srgb2lin(b);
  const X = R*0.4124564 + G*0.3575761 + B*0.1804375;
  const Y = R*0.2126729 + G*0.7151522 + B*0.0721750;
  const Z = R*0.0193339 + G*0.1191920 + B*0.9503041;
  const xn=0.95047, yn=1.00000, zn=1.08883;
  const f = (t)=> t>0.008856?Math.cbrt(t):(7.787*t+16/116);
  const fx=f(X/xn), fy=f(Y/yn), fz=f(Z/zn);
  const L = 116*fy - 16;
  const a = 500*(fx-fy);
  const b2 = 200*(fy-fz);
  return {L,a,b:b2};
}

// -------- Blur/pose helpers
function varianceOfLaplacian(imageData) {
  const W=imageData.width, H=imageData.height, d=imageData.data;
  const gray = new Float32Array(W*H);
  for (let y=0,i=0;y<H;y++)
    for (let x=0;x<W;x++,i+=4) {
      gray[(y*W)+x] = 0.299*d[i]+0.587*d[i+1]+0.114*d[i+2];
    }
  
  const k = [[0,1,0],[1,-4,1],[0,1,0]];
  let sum=0, sumSq=0, n=0;
  for (let y=1;y<H-1;y++){
    for(let x=1;x<W-1;x++){
      let v=0;
      for (let j=-1;j<=1;j++)
        for (let i=-1;i<=1;i++)
          v += k[j+1][i+1] * gray[(y+j)*W + (x+i)];
      sum += v; sumSq += v*v; n++;
    }
  }
  const mean = sum/n;
  return sumSq/n - mean*mean;
}

function centers(pts, idxs) {
  const xs = idxs.map(i => pts[i].x), ys = idxs.map(i => pts[i].y);
  return { x: xs.reduce((a,b)=>a+b,0)/xs.length, y: ys.reduce((a,b)=>a+b,0)/ys.length };
}

function eyeCenters(pts){
  const L = [36,37,38,39,40,41].map(i=>pts[i]);
  const R = [42,43,44,45,46,47].map(i=>pts[i]);
  const avg = (arr, key)=> arr.reduce((s,p)=>s+p[key],0)/arr.length;
  return {
    left:{x:avg(L,'x'),y:avg(L,'y')},
    right:{x:avg(R,'x'),y:avg(R,'y')}
  };
}

function poseFromEyesNose(pts){
  const {left,right} = eyeCenters(pts);
  const nose = pts[33];
  const roll = Math.atan2(right.y-left.y, right.x-left.x) * 180/Math.PI;
  const dl = dist(nose,left), dr = dist(nose,right);
  const yaw = Math.atan((dr - dl) / ((dr+dl)/2)) * 180/Math.PI;
  const mouthTop = pts[51];
  const eyeY = (left.y + right.y)/2;
  const pitch = Math.atan((mouthTop.y - eyeY) / Math.max(1, right.x-left.x)) * 180/Math.PI;
  return { roll, yaw, pitch };
}

// Alignment + illumination
function alignToCanonical(srcCanvas, pts, W=400, H=480) {
  const {left,right} = eyeCenters(pts);
  const eyeDist = dist(left,right);
  if (!Number.isFinite(eyeDist) || eyeDist < 10) return { canvas: srcCanvas, tx: null };
  
  const targetLeft = { x: W*0.35, y: H*0.38 };
  const targetRight = { x: W*0.65, y: H*0.38 };
  const s = dist(targetLeft, targetRight) / eyeDist;
  const angle = Math.atan2(targetRight.y - targetLeft.y, targetRight.x - targetLeft.x) - 
                Math.atan2(right.y - left.y, right.x - left.x);
  
  const out = createCanvas(W,H);
  const ctx = out.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.translate(targetLeft.x, targetLeft.y);
  ctx.rotate(angle);
  ctx.scale(s, s);
  ctx.translate(-left.x, -left.y);
  ctx.drawImage(srcCanvas, 0, 0);
  
  return { canvas: out, tx: { s, angle, ref:left, targetLeft, W, H } };
}

function normalizeIllumination(canvas) {
  const W=canvas.width, H=canvas.height;
  const ctx = canvas.getContext('2d');
  const img = ctx.getImageData(0,0,W,H);
  const d = img.data;
  
  const gamma = 0.9;
  const g = (u)=> {
    const x = u/255;
    return Math.min(255, Math.max(0, Math.pow(x, gamma)*255));
  };
  
  const lum = new Float32Array(W*H);
  for (let y=0,i=0;y<H;y++)
    for (let x=0;x<W;x++,i+=4) {
      lum[(y*W)+x] = 0.299*d[i]+0.587*d[i+1]+0.114*d[i+2];
    }
  
  const blur = new Float32Array(W*H);
  for (let y=0;y<H;y++){
    for(let x=0;x<W;x++){
      let sum=0,cnt=0;
      for (let j=-2;j<=2;j++){
        for(let i=-2;i<=2;i++){
          const xx = clamp(x+i,0,W-1), yy=clamp(y+j,0,H-1);
          sum += lum[yy*W+xx]; cnt++;
        }
      }
      blur[y*W+x] = sum/cnt;
    }
  }
  
  for (let y=0,i=0;y<H;y++){
    for(let x=0;x<W;x++,i+=4){
      const L = lum[y*W+x];
      const Ls = clamp(L*1.1 - blur[y*W+x]*0.1, 0, 255);
      const r = g(d[i]), g2 = g(d[i+1]), b = g(d[i+2]);
      const curLum = 0.299*r+0.587*g2+0.114*b;
      const scale = curLum>1 ? (Ls/curLum) : 1;
      d[i] = clamp(r*scale, 0, 255);
      d[i+1] = clamp(g2*scale, 0, 255);
      d[i+2] = clamp(b*scale, 0, 255);
    }
  }
  ctx.putImageData(img,0,0);
  return canvas;
}

// ============================================================================
// 6) Feature scoring (wrinkles/texture/pigment/etc.)
// ============================================================================
function foreheadWrinkleScore(canvas, landmarks) {
  const ctx = canvas.getContext('2d');
  const pts = pointsFromLandmarks(landmarks);
  if (!pts.length) return 0;
  
  const left = Math.round(pts[17].x);
  const right = Math.round(pts[26].x);
  const browY = Math.min(pts[19].y, pts[24].y);
  const topY = Math.round(browY - 32);
  const rect = roiClampRect(canvas, left, topY, right-left, 34);
  const roi = getROI(ctx, rect);
  const d1 = edgeDensity(roi, 48);
  
  const small = createCanvas(Math.max(16, Math.round(rect.w/2)), Math.max(8, Math.round(rect.h/2)));
  small.getContext('2d').drawImage(canvas, rect.x, rect.y, rect.w, rect.h, 0,0,small.width, small.height);
  const d2 = edgeDensity(small.getContext('2d').getImageData(0,0,small.width, small.height), 48);
  
  return clamp((d1*0.6 + d2*0.4) * 900, 0, 100);
}

function glabellarScore(canvas, landmarks) {
  const ctx = canvas.getContext('2d');
  const pts = pointsFromLandmarks(landmarks);
  if (!pts.length) return 0;
  
  const midX = pts[27].x;
  const yTop = pts[21].y - 10;
  const yBot = pts[33].y - 8;
  const rect = roiClampRect(canvas, midX - 14, yTop, 28, Math.max(18, yBot - yTop));
  const d = edgeDensity(getROI(ctx, rect), 52);
  return clamp(d * 1100, 0, 100);
}

function crowsFeetScore(canvas, landmarks) {
  const ctx = canvas.getContext('2d');
  const pts = pointsFromLandmarks(landmarks);
  if (!pts.length) return 0;
  
  const leftOuter = pts[39];
  const rightOuter = pts[42];
  const rectL = roiClampRect(canvas, leftOuter.x + 8, leftOuter.y - 10, 26, 22);
  const rectR = roiClampRect(canvas, rightOuter.x - 34, rightOuter.y - 10, 26, 22);
  const d = (edgeDensity(getROI(ctx, rectL), 52) + edgeDensity(getROI(ctx, rectR), 52)) / 2;
  return clamp(d * 1100, 0, 100);
}

function darkCircleScore(canvas, landmarks) {
  const pts = pointsFromLandmarks(landmarks);
  if (!pts.length) return 0;
  
  const leftEye = pts.slice(36, 42);
  const rightEye = pts.slice(42, 48);
  
  const sample = (region, dy) => {
    let sum = 0, n = 0;
    for (const p of region) {
      const x = clamp(Math.round(p.x), 0, canvas.width - 1);
      const y = clamp(Math.round(p.y + dy), 0, canvas.height - 1);
      sum += luminanceAt(canvas, x, y);
      n++;
    }
    return n ? sum / n : 0;
  };
  
  const under = (sample(leftEye, 12) + sample(rightEye, 12)) / 2;
  const cheek = (sample(leftEye, 30) + sample(rightEye, 30)) / 2;
  const diff = Math.max(0, cheek - under);
  return Math.min(100, (diff / 35) * 100);
}

function eyebagPuffinessScore(canvas, landmarks) {
  const pts = pointsFromLandmarks(landmarks);
  if (!pts.length) return 0;
  const ctx = canvas.getContext('2d');
  
  const lEye = centers(pts, [36,37,38,39,40,41]);
  const rEye = centers(pts, [42,43,44,45,46,47]);
  const bandH = 14;
  const bandW = Math.abs(pts[39].x - pts[36].x);
  
  const lRect = roiClampRect(canvas, lEye.x - bandW/2, lEye.y + 6, bandW, bandH);
  const rRect = roiClampRect(canvas, rEye.x - bandW/2, rEye.y + 6, bandW, bandH);
  
  const dL = edgeDensity(getROI(ctx, lRect), 40);
  const dR = edgeDensity(getROI(ctx, rRect), 40);
  return clamp(((dL + dR) / 2) * 850, 0, 100);
}

function saggingScore(landmarks) {
  const pts = pointsFromLandmarks(landmarks);
  if (!pts.length) return 0;
  
  const le = pts[36], re = pts[45];
  const lm = pts[48], rm = pts[54];
  const eyeY = (le.y + re.y) / 2;
  const mouthY = (lm.y + rm.y) / 2;
  const delta = mouthY - eyeY;
  return clamp((delta / 55) * 100, 0, 100);
}

function cheekRects(canvas, pts) {
  const lEye = centers(pts, [36,37,38,39,40,41]);
  const rEye = centers(pts, [42,43,44,45,46,47]);
  const mouthTop = centers(pts, [51,52,53]);
  const noseTip = pts[33];
  const jawL = pts[3];
  const jawR = pts[13];
  
  const yTop = Math.min(lEye.y, rEye.y) + 8;
  const yBot = Math.max(mouthTop.y, noseTip.y) + 12;
  
  const leftRect = roiClampRect(canvas, jawL.x, yTop, (noseTip.x - jawL.x) * 0.9, (yBot - yTop) * 1.0);
  const rightRect = roiClampRect(canvas, noseTip.x + 2, yTop, (jawR.x - noseTip.x) * 0.9, (yBot - yTop) * 1.0);
  return { leftRect, rightRect };
}

function rednessScore(canvas, landmarks) {
  const pts = pointsFromLandmarks(landmarks);
  if (!pts.length) return 0;
  const ctx = canvas.getContext('2d');
  
  const { leftRect, rightRect } = cheekRects(canvas, pts);
  const lLab = rgbToLabAvg(getROI(ctx, leftRect));
  const rLab = rgbToLabAvg(getROI(ctx, rightRect));
  const aMean = (lLab.a + rLab.a) / 2;
  return clamp(((aMean - 15) / 20) * 100, 0, 100);
}

function pigmentationScore(canvas, landmarks) {
  const pts = pointsFromLandmarks(landmarks);
  if (!pts.length) return 0;
  const ctx = canvas.getContext('2d');
  
  const { leftRect, rightRect } = cheekRects(canvas, pts);
  const l = getROI(ctx, leftRect);
  const r = getROI(ctx, rightRect);
  const lv = meanStdRGB(l).lStd;
  const rv = meanStdRGB(r).lStd;
  const cv = (lv + rv) / 2;
  return clamp(((cv - 6) / 20) * 100, 0, 100);
}

function textureScore(canvas, landmarks) {
  const pts = pointsFromLandmarks(landmarks);
  if (!pts.length) return 0;
  const ctx = canvas.getContext('2d');
  
  const { leftRect, rightRect } = cheekRects(canvas, pts);
  const dL = edgeDensity(getROI(ctx, leftRect), 42);
  const dR = edgeDensity(getROI(ctx, rightRect), 42);
  return clamp(((dL + dR) / 2) * 1100, 0, 100);
}

function perioralFineLinesScore(canvas, landmarks) {
  const ctx = canvas.getContext('2d');
  const pts = pointsFromLandmarks(landmarks);
  if (!pts.length) return 0;
  
  const left = pts[48], right = pts[54], top = pts[51], bot = pts[57];
  const rect = roiClampRect(canvas, left.x - 12, top.y - 12, (right.x - left.x) + 24, (bot.y - top.y) + 28);
  const d = edgeDensity(getROI(ctx, rect), 52);
  return clamp(d * 1100, 0, 100);
}

function thinLipsScore(landmarks) {
  const pts = pointsFromLandmarks(landmarks);
  if (!pts.length) return 0;
  
  const width = dist(pts[48], pts[54]);
  const height = dist(pts[51], pts[57]);
  const ratio = height / Math.max(1, width);
  return clamp(((0.22 - ratio) / 0.12) * 100, 0, 100);
}

function marionetteScore(canvas, landmarks) {
  const ctx = canvas.getContext('2d');
  const pts = pointsFromLandmarks(landmarks);
  if (!pts.length) return 0;
  
  const left = pts[48], right = pts[54], chin = pts[8];
  const rectL = roiClampRect(canvas, left.x - 10, left.y, 24, Math.max(16, chin.y - left.y));
  const rectR = roiClampRect(canvas, right.x - 14, right.y, 24, Math.max(16, chin.y - right.y));
  const d = (edgeDensity(getROI(ctx, rectL), 52) + edgeDensity(getROI(ctx, rectR), 52)) / 2;
  return clamp(d * 1150, 0, 100);
}

function weakChinScore(landmarks) {
  const pts = pointsFromLandmarks(landmarks);
  if (!pts.length) return 0;
  
  const lipToChin = Math.max(1, pts[8].y - pts[57].y);
  const browToChin = Math.max(1, pts[8].y - Math.min(pts[19].y, pts[24].y));
  const ratio = lipToChin / browToChin;
  return clamp(((0.20 - ratio) / 0.10) * 100, 0, 100);
}

function jawlineSoftnessScore(canvas, landmarks) {
  const ctx = canvas.getContext('2d');
  const pts = pointsFromLandmarks(landmarks);
  if (!pts.length) return 0;
  
  const angL = pts[5], angR = pts[11];
  const rectL = roiClampRect(canvas, angL.x - 18, angL.y - 18, 36, 36);
  const rectR = roiClampRect(canvas, angR.x - 18, angR.y - 18, 36, 36);
  const d = (edgeDensity(getROI(ctx, rectL), 48) + edgeDensity(getROI(ctx, rectR), 48)) / 2;
  return clamp(100 - d * 1200, 0, 100);
}

function doubleChinScore(canvas, landmarks) {
  const ctx = canvas.getContext('2d');
  const pts = pointsFromLandmarks(landmarks);
  if (!pts.length) return 0;
  
  const chin = pts[8];
  const subRect = roiClampRect(canvas, chin.x - 40, chin.y + 4, 80, 34);
  const subStats = meanStdRGB(getROI(ctx, subRect));
  
  const lEye = centers(pts, [36,37,38,39,40,41]);
  const rEye = centers(pts, [42,43,44,45,46,47]);
  const cheeksLum = (()=>{
    const left = roiClampRect(canvas, lEye.x-30, lEye.y+10, 40, 35);
    const right= roiClampRect(canvas, rEye.x-10, rEye.y+10, 40, 35);
    const l=meanStdRGB(getROI(ctx,left)).lMean;
    const r=meanStdRGB(getROI(ctx,right)).lMean;
    return (l+r)/2;
  })();
  
  const jawSoft = jawlineSoftnessScore(canvas, landmarks);
  const lumDiff = clamp((cheeksLum - subStats.lMean) / 40 * 100, 0, 100);
  return clamp(0.6 * lumDiff + 0.4 * jawSoft, 0, 100);
}

function templeHollownessScore(canvas, landmarks) {
  const ctx = canvas.getContext('2d');
  const pts = pointsFromLandmarks(landmarks);
  if (!pts.length) return 0;
  
  const leftT = pts[17], rightT = pts[26];
  const rectL = roiClampRect(canvas, leftT.x - 26, leftT.y - 20, 24, 40);
  const rectR = roiClampRect(canvas, rightT.x + 2, rightT.y - 20, 24, 40);
  
  const l = meanStdRGB(getROI(ctx, rectL));
  const r = meanStdRGB(getROI(ctx, rectR));
  const templeLum = (l.lMean + r.lMean) / 2;
  
  const cheekLum = (() => {
    const lEye = centers(pts, [36,37,38,39,40,41]);
    const rEye = centers(pts, [42,43,44,45,46,47]);
    const m = centers(pts, [51,52,53]);
    const yTop = Math.min(lEye.y, rEye.y) + 10;
    const yBot = Math.max(m.y, pts[33].y) + 10;
    const Lr = roiClampRect(canvas, pts[3].x, yTop, (pts[33].x - pts[3].x) * 0.8, (yBot - yTop));
    const Rr = roiClampRect(canvas, pts[33].x, yTop, (pts[13].x - pts[33].x) * 0.8, (yBot - yTop));
    const lm=meanStdRGB(getROI(ctx, Lr)).lMean;
    const rm=meanStdRGB(getROI(ctx, Rr)).lMean;
    return (lm+rm)/2;
  })();
  
  const diff = clamp((cheekLum - templeLum) / 35 * 100, 0, 100);
  return diff;
}

function dullSkinScore(canvas, landmarks) {
  const ctx = canvas.getContext('2d');
  const pts = pointsFromLandmarks(landmarks);
  if (!pts.length) return 50;
  
  const jawL = pts[3], jawR = pts[13], browL = pts[19], browR = pts[24];
  const x = Math.min(jawL.x, browL.x);
  const y = Math.min(browL.y, browR.y) - 20;
  const w = Math.max(jawR.x - x, 40);
  const h = Math.max(pts[8].y - y, 40);
  const rect = roiClampRect(canvas, x, y, w, h);
  const stats = meanStdRGB(getROI(ctx, rect));
  return clamp((150 - stats.lMean) / 100 * 100, 0, 100);
}

// ============================================================================
// 7) Symmetry (IPD-normalized mm) with near-frontal gate
// ============================================================================
const symmetricPairs = [
  [0,16],[1,15],[2,14],[3,13],[4,12],[5,11],[6,10],[7,9],
  [17,26],[18,25],[19,24],[20,23],[21,22],
  [36,45],[37,44],[38,43],[39,42],[40,47],[41,46],
  [31,35],[32,34],
  [48,54],[49,53],[50,52],[60,64],[61,63],[67,65]
];

function symmetryStats(alignedCanvas, pts){
  if (!pts.length) return null;
  const {left,right} = eyeCenters(pts);
  const x0 = (left.x + right.x)/2;
  const ipdPx = dist(left, right);
  if (!Number.isFinite(ipdPx) || ipdPx < 10) return null;
  
  const diffs = [];
  const regionMax = { jaw:0, cheek:0, brow:0, mouth:0, eye:0, nose:0 };
  for (const [iL,iR] of symmetricPairs){
    const L = pts[iL], R = pts[iR];
    const Rm = { x: 2*x0 - R.x, y: R.y };
    const dpx = Math.hypot(L.x - Rm.x, L.y - Rm.y);
    diffs.push(dpx);
    
    const bucket = (iL<=8 || (iR>=8 && iL<=16)) ? 'jaw' :
                  (iL>=17 && iL<=26) ? 'brow' :
                  (iL>=36 && iL<=47) ? 'eye' :
                  (iL>=31 && iL<=35) ? 'nose' :
                  (iL>=48 && iL<=67) ? 'mouth' : 'cheek';
    regionMax[bucket] = Math.max(regionMax[bucket], dpx);
  }
  
  const meanSq = diffs.reduce((s,v)=>s+v*v,0) / diffs.length;
  const rmsPx = Math.sqrt(meanSq);
  const mmPerPx = ASSUMED_IPD_MM / ipdPx;
  const rmsMm = rmsPx * mmPerPx;
  const pctIPD = (rmsPx / ipdPx);
  
  let largest = {region:'jaw', mm: regionMax.jaw*mmPerPx};
  for (const k of Object.keys(regionMax)) {
    const mm = regionMax[k]*mmPerPx;
    if (mm > largest.mm) largest = {region:k, mm};
  }
  
  const bucket = pctIPD*100 < 2 ? 'Low' : pctIPD*100 < 4 ? 'Moderate' : 'High';
  return { ipdPx, rmsPx, rmsMm, pctIPD, largestRegion: largest.region, largestRegionMm: largest.mm, bucket };
}

// ============================================================================
// 8) Treatments & mapping (top 2–3 only, tighter gates)
// ============================================================================
const FALLBACK_TREATMENTS = [
  { id: 'botox', name: 'Botox / Dysport', description: 'Soften dynamic forehead, glabella & crow’s feet.', durationMin: 15, priceMin: 150, priceMax: 280 },
  { id: 'fillers-teartrough', name: 'Tear trough filler', description: 'Reduce hollowness and dark under-eye shadows.', durationMin: 20, priceMin: 280, priceMax: 350 },
  { id: 'fillers-midface', name: 'Midface/Cheek filler', description: 'Restore midface volume and lift.', durationMin: 25, priceMin: 250, priceMax: 400 },
  { id: 'fillers-temple', name: 'Temple filler', description: 'Replete temple volume for youthful contour.', durationMin: 20, priceMin: 280, priceMax: 380 },
  { id: 'fillers-lip', name: 'Lip filler', description: 'Add volume/definition and correct asymmetries.', durationMin: 20, priceMin: 200, priceMax: 320 },
  { id: 'fillers-marionette', name: 'Marionette line filler', description: 'Lift corners and soften lines.', durationMin: 20, priceMin: 220, priceMax: 320 },
  { id: 'fillers-chin', name: 'Chin augmentation (filler)', description: 'Improve chin projection and balance.', durationMin: 20, priceMin: 250, priceMax: 400 },
  { id: 'fillers-jawline', name: 'Jawline contouring (filler)', description: 'Define mandibular angle and jawline.', durationMin: 25, priceMin: 300, priceMax: 480 },
  { id: 'profhilo', name: 'Biostimulators (Profhilo/Jalupro/Karisma)', description: 'Collagen/elastin stimulation for overall quality.', durationMin: 20, priceMin: 220, priceMax: 300 },
  { id: 'hifu', name: 'HIFU / RF tightening', description: 'Non-invasive tightening to improve laxity.', durationMin: 30, priceMin: 100, priceMax: 250 },
  { id: 'fillers-fine', name: 'Fine-line filler', description: 'Etched lines (perioral etc.).', durationMin: 15, priceMin: 180, priceMax: 250 },
  { id: 'ipl', name: 'IPL / Vascular laser', description: 'Target redness and broken capillaries.', durationMin: 20, priceMin: 180, priceMax: 300 },
  { id: 'chemical-peel', name: 'Chemical peel', description: 'Brighten and even pigmentation/dullness.', durationMin: 20, priceMin: 120, priceMax: 220 },
  { id: 'microneedling', name: 'Microneedling / Resurfacing', description: 'Refine texture and acne scarring.', durationMin: 30, priceMin: 150, priceMax: 280 },
  { id: 'skinbooster', name: 'Skin booster (HA)', description: 'Hydration & glow for crepey/fine texture.', durationMin: 20, priceMin: 180, priceMax: 260 },
  { id: 'prp', name: 'PRP / Mesotherapy (eye/face)', description: 'Improve under-eye quality and fine lines.', durationMin: 25, priceMin: 150, priceMax: 260 },
  { id: 'lipolysis', name: 'Fat-dissolve (submental) / HIFU', description: 'Reduce double chin fullness (non-surgical).', durationMin: 25, priceMin: 200, priceMax: 450 },
  { id: 'laser-hair', name: 'Laser hair removal (face)', description: 'Reduce excess facial hair over sessions.', durationMin: 15, priceMin: 60, priceMax: 150 },
];

function loadTreatments() {
  const t = readJSON(TREATMENTS_DB, null);
  if (!t || !Array.isArray(t.items)) return { items: FALLBACK_TREATMENTS };
  const ids = new Set(t.items.map(x => x.id));
  const missing = FALLBACK_TREATMENTS.filter(x => !ids.has(x.id));
  return { items: [...t.items, ...missing] };
}

function severityBucket(v) {
  if (v >= 65) return 'severe';
  if (v >= 40) return 'moderate';
  if (v >= 25) return 'mild';
  return 'none';
}

function mapToTreatmentIds(f) {
  const featList = [
    ['forehead', f.foreheadWrinkleScore],
    ['glabella', f.glabellarScore],
    ['crows', f.crowsFeetScore],
    ['underEyeDark', f.darkCircleScore],
    ['eyebag', f.eyebagScore],
    ['temple', f.templeHollownessScore],
    ['sag', f.saggingScore],
    ['naso', f.nasolabialScore],
    ['perioral', f.perioralFineLinesScore],
    ['thinLips', f.thinLipsScore],
    ['marionette', f.marionetteScore],
    ['weakChin', f.weakChinScore],
    ['jawSoft', f.jawlineSoftnessScore],
    ['doubleChin', f.doubleChinScore],
    ['red', f.rednessScore],
    ['pig', f.pigmentationScore],
    ['tex', f.textureScore],
    ['dull', f.dullSkinScore],
  ].filter(([_k,v]) => v >= 30);
  
  featList.sort((a,b)=>b[1]-a[1]);
  const seeds = featList.slice(0, 5);
  const picks = [];
  
  const add = (id, w) => {
    const cur = picks.find(p => p.id === id);
    if (!cur) picks.push({ id, w });
    else cur.w = Math.max(cur.w, w);
  };
  
  for (const [key, v] of seeds) {
    const sev = severityBucket(v);
    if (sev === 'none' || sev === 'mild') continue;
    
    switch (key) {
      case 'forehead': case 'glabella': case 'crows':
        add('botox', v + 5);
        if (key === 'crows' && v >= 55) add('skinbooster', v - 5);
        break;
      case 'underEyeDark':
        if (f.eyebagScore < 40) add('fillers-teartrough', v + 8);
        else add('prp', v + 3);
        break;
      case 'eyebag':
        add('prp', v + 4);
        if (v >= 55) add('skinbooster', v - 6);
        break;
      case 'temple':
        add('fillers-temple', v + 5);
        break;
      case 'sag': case 'naso':
        add('fillers-midface', v + 6);
        if (v >= 50) add('hifu', v);
        break;
      case 'perioral':
        add('fillers-fine', v + 3);
        if (v >= 50) add('skinbooster', v - 8);
        break;
      case 'thinLips':
        add('fillers-lip', v + 8);
        break;
      case 'marionette':
        add('fillers-marionette', v + 6);
        break;
      case 'weakChin':
        add('fillers-chin', v + 6);
        break;
      case 'jawSoft':
        add('fillers-jawline', v + 6);
        if (f.saggingScore >= 50) add('hifu', v - 4);
        break;
      case 'doubleChin':
        add('lipolysis', v + 6);
        if (f.saggingScore >= 50) add('hifu', v - 6);
        break;
      case 'red':
        add('ipl', v + 5);
        break;
      case 'pig':
        add('chemical-peel', v + 6);
        if (v >= 55) add('ipl', v - 6);
        break;
      case 'tex':
        add('microneedling', v + 6);
        if (v >= 50) add('skinbooster', v - 6);
        break;
      case 'dull':
        add('skinbooster', v + 5);
        add('chemical-peel', v - 6);
        break;
    }
  }
  
  if ((f.ageEstimate >= 35 || f.textureScore >= 55 || f.saggingScore >= 55) && !picks.find(p=>p.id==='profhilo')) {
    picks.push({ id: 'profhilo', w: Math.max(f.ageEstimate, f.textureScore, f.saggingScore) - 5 });
  }
  
  picks.sort((a,b)=>b.w-a.w);
  const unique = [];
  for (const p of picks) if (!unique.find(u => u.id === p.id)) unique.push(p);
  return unique.slice(0, 3).map(p => p.id);
}

// ============================================================================
// 9) Narrative builder (English, includes symmetry & age range)
// ============================================================================
function buildEnglishNarrative(f, suggestionCards, disclaimer) {
  const obs = [
    ['Forehead lines', f.foreheadWrinkleScore],
    ['Glabellar (frown) lines', f.glabellarScore],
    ['Crow’s feet', f.crowsFeetScore],
    ['Under-eye darkness', f.darkCircleScore],
    ['Eye-bag puffiness', f.eyebagScore],
    ['Temple hollowing', f.templeHollownessScore],
    ['Sagging/laxity', f.saggingScore],
    ['Nasolabial folds', f.nasolabialScore],
    ['Perioral fine lines', f.perioralFineLinesScore],
    ['Thin lips', f.thinLipsScore],
    ['Marionette lines', f.marionetteScore],
    ['Weak chin', f.weakChinScore],
    ['Jawline softness', f.jawlineSoftnessScore],
    ['Double-chin fullness', f.doubleChinScore],
    ['Redness', f.rednessScore],
    ['Pigmentation', f.pigmentationScore],
    ['Texture/pores/scarring', f.textureScore],
    ['Dullness', f.dullSkinScore],
  ].sort((a,b)=>b[1]-a[1]).filter(([,v])=>v>=25).slice(0,8).map(([k])=>k);
  
  const symTxt = (()=>{
    if (!Number.isFinite(f.symRmsMm)) return '—';
    const pct = (f.symPctIPD*100).toFixed(1);
    return `${f.symRmsMm.toFixed(1)} mm (${pct}% IPD), ${f.symBucket}`;
  })();
  
  const ageTxt = Number.isFinite(f.ageEstimate) ? `~${f.ageEstimate} (${f.ageLow}–${f.ageHigh})` : '—';
  
  const lines = [
    `Here’s a concise, non-medical summary based on your photos:`,
    `• Age estimate: ${ageTxt}`,
    `• Dominant expression: ${f.topEmotion || '-'}`,
    `• Face asymmetry: ${symTxt}`,
    `• Key observations: ${obs.join('; ')}.`,
    ``,
    `Top suggested treatments:`,
    ...suggestionCards.map((c, i) => `${i+1}. ${c.name} — ${c.description}`),
    ``,
    disclaimer,
  ];
  return lines.join('\n');
}

// ============================================================================
// 10) UPDATED: detection chain + age picking helpers (Face-API only)
// ============================================================================
// Always grab Face-API age/expressions.
async function detectOne(faceapi, canvas) {
  if (!canvas) return null;
  let chain;
  if (USE_TINY && faceapi.nets.tinyFaceDetector) {
    const TinyOpts = faceapi.TinyFaceDetectorOptions || (class { constructor(o){Object.assign(this,o)} });
    const opts = new TinyOpts({ inputSize: 416, scoreThreshold: 0.4 });
    chain = faceapi.detectSingleFace(canvas, opts);
  } else {
    chain = faceapi.detectSingleFace(canvas);
  }
  return await chain.withFaceLandmarks(true).withAgeAndGender().withFaceExpressions();
}

function isSaneAge(a) {
  return Number.isFinite(a) && a >= 5 && a <= 100;
}

function pickAge(_onnxAge, faAge) {
  // ONNX removed; just use Face-API age if sane
  if (isSaneAge(faAge)) return faAge;
  return NaN;
}

const aggMean = (arr)=> {
  const a = arr.filter((v)=>Number.isFinite(v));
  if (!a.length) return 0;
  return a.reduce((s,v)=>s+v,0)/a.length;
};

const aggStd = (arr)=>{
  const a = arr.filter((v)=>Number.isFinite(v));
  if (a.length<2) return 0;
  const m = aggMean(a);
  const v = a.reduce((s,v)=>s+(v-m)*(v-m),0)/a.length;
  return Math.sqrt(Math.max(0,v));
};

async function analyzeThree(images) {
  await loadModelsOnce();
  // Try to make FaceMesh available; non-fatal if it fails
  await loadFaceMeshOnce();
  const faceapi = await getFaceApi();
  
  const rawCanvas = {};
  for (const k of ['front', 'left', 'right']) {
    const val = images[k];
    if (!val) throw new Error(`MISSING_IMAGE_${k.toUpperCase()}`);
    // NEW: allow Buffer OR path
    rawCanvas[k] = await loadImageAsCanvasFromAny(val);
  }

  
  // Detect on all views (Face-API age/gender/expressions)
  const dets = {};
  for (const k of ['front', 'left', 'right']) dets[k] = await detectOne(faceapi, rawCanvas[k]);
  
  const any = ['front', 'left', 'right'].find(k => !!dets[k]);
  if (!any) throw new Error('NO_FACE_DETECTED');
  
  const perView = {};
  const ageViews = [];
  const exprViews = [];
  
  for (const k of ['front', 'left', 'right']) {
    const d = dets[k];
    if (!d) continue;
    
    const pts = pointsFromLandmarks(d.landmarks);
    // Log the number of landmarks and the first few points
    console.log(`Landmarks for ${k}:`, pts.length); // Log the number of landmarks
    console.log(`First 5 landmarks for ${k}:`, pts.slice(0, 5)); // Log first 5 landmarks
    
    const { canvas: aligned } = alignToCanonical(rawCanvas[k], pts, 400, 480);
    normalizeIllumination(aligned);
    const ctxA = aligned.getContext('2d');
    const blurVar = varianceOfLaplacian(ctxA.getImageData(0, 0, aligned.width, aligned.height));
    const pose = poseFromEyesNose(pts);
    const nearFrontal = Math.abs(pose.roll) <= 8 && Math.abs(pose.yaw) <= 8;
    
    // Feature scores (68-landmarks-based)
    const forehead = foreheadWrinkleScore(aligned, d.landmarks);
    const glabella = glabellarScore(aligned, d.landmarks);
    const crows = crowsFeetScore(aligned, d.landmarks);
    const dark = darkCircleScore(aligned, d.landmarks);
    const bag = eyebagPuffinessScore(aligned, d.landmarks);
    const sag = saggingScore(d.landmarks);
    const red = rednessScore(aligned, d.landmarks);
    const tex = textureScore(aligned, d.landmarks);
    const pig = pigmentationScore(aligned, d.landmarks);
    const perioral = perioralFineLinesScore(aligned, d.landmarks);
    const thin = thinLipsScore(d.landmarks);
    const mar = marionetteScore(aligned, d.landmarks);
    const chin = weakChinScore(d.landmarks);
    const jawSoft = jawlineSoftnessScore(aligned, d.landmarks);
    const dbl = doubleChinScore(aligned, d.landmarks);
    const temple = templeHollownessScore(aligned, d.landmarks);
    const dull = dullSkinScore(aligned, d.landmarks);
    
    // Face-API age/gender/expressions (no ONNX)
    const faAge = Number.isFinite(d.age) ? d.age : NaN;
    const ageEst = pickAge(NaN, faAge);
    const gender = d.gender || 'unknown';
    const expr = d.expressions || {};
    
    // symmetry per view if near-frontal
    let sym = null;
    if (nearFrontal) {
      const ptsAligned = pointsFromLandmarks(d.landmarks);
      sym = symmetryStats(aligned, ptsAligned);
    }
    
    // ---- NEW: run FaceMesh on aligned canvas, capture 468/478 points (optional)
    let mesh468 = null;
    if (FACE_MESH_ENABLE && MESH_READY) {
      const meshRes = await estimateMeshOnCanvas(aligned);
      if (meshRes && meshRes.points && meshRes.points.length >= 468) {
        mesh468 = { count: meshRes.count, points: meshRes.points }; // keep all points (468 or 478)
      }
    }
    
    perView[k] = {
      age: ageEst, expr, gender, blurVar, pose, nearFrontal, aligned,
      forehead, glabella, crows, dark, bag, sag, red, tex, pig, perioral, thin, mar, chin, jawSoft, dbl, temple, dull,
      symmetry: sym,
      // NEW: attach mesh (if available)
      mesh468,
    };
    
    if (isSaneAge(ageEst)) ageViews.push(ageEst);
    exprViews.push(expr);
  }
  
  // Aggregate across views
  const meanOf = (key) => aggMean(Object.values(perView).map(v => v[key]));
  
  // Emotions aggregation via Face-API
  let topEmotion = '-';
  {
    const keys = new Set();
    exprViews.forEach(e => Object.keys(e || {}).forEach(k => keys.add(k)));
    const avg = {};
    keys.forEach(k => { avg[k] = aggMean(exprViews.map(e => (e?.[k] ?? 0))); });
    topEmotion = Object.entries(avg).sort((a, b) => b[1] - a[1])[0]?.[0] || '-';
  }
  
  // symmetry aggregate
  const symMmArr = Object.values(perView).map(v => v.symmetry?.rmsMm).filter(Number.isFinite);
  const symMm = aggMean(symMmArr);
  const symStd = aggStd(symMmArr);
  const symPctArr = Object.values(perView).map(v => v.symmetry?.pctIPD).filter(Number.isFinite);
  const symPct = aggMean(symPctArr);
  const symBucket = (() => {
    if (!Number.isFinite(symPct)) return '—';
    const p = symPct * 100;
    return p < 2 ? 'Low' : p < 4 ? 'Moderate' : 'High';
  })();
  
  // age smoothing + range (robust to missing)
  const rawAges = ageViews.filter(isSaneAge);
  let ageMean = rawAges.length ? Math.round(aggMean(rawAges)) : NaN;
  let ageLow = Number.isFinite(ageMean) ? ageMean - 3 : null;
  let ageHigh = Number.isFinite(ageMean) ? ageMean + 3 : null;
  
  const blurAvg = aggMean(Object.values(perView).map(v => v.blurVar));
  if (Number.isFinite(ageMean) && blurAvg < 35) {
    ageLow = ageMean - 4;
    ageHigh = ageMean + 4;
  }
  
  const findings = {
    viewUsed: any,
    ageEstimate: ageMean,
    ageLow, ageHigh,
    gender: (Object.values(perView).find(v => v.gender && v.gender !== 'unknown')?.gender) || 'unknown',
    topEmotion,
    symmetry: Number.isFinite(symPct) ? Math.max(0, Math.min(100, Math.round(100 - symPct * 100))) : 0,
    symRmsMm: Number.isFinite(symMm) ? Number(symMm.toFixed(2)) : NaN,
    symPctIPD: Number.isFinite(symPct) ? Number(symPct.toFixed(4)) : NaN,
    symBucket,
    symStdMm: Number.isFinite(symStd) ? Number(symStd.toFixed(2)) : NaN,
    
    foreheadWrinkleScore: round(meanOf('forehead')),
    glabellarScore: round(meanOf('glabella')),
    crowsFeetScore: round(meanOf('crows')),
    darkCircleScore: round(meanOf('dark')),
    eyebagScore: round(meanOf('bag')),
    saggingScore: round(meanOf('sag')),
    rednessScore: round(meanOf('red')),
    textureScore: round(meanOf('tex')),
    pigmentationScore: round(meanOf('pig')),
    perioralFineLinesScore: round(meanOf('perioral')),
    thinLipsScore: round(meanOf('thin')),
    marionetteScore: round(meanOf('mar')),
    weakChinScore: round(meanOf('chin')),
    jawlineSoftnessScore: round(meanOf('jawSoft')),
    doubleChinScore: round(meanOf('dbl')),
    templeHollownessScore: round(meanOf('temple')),
    dullSkinScore: round(meanOf('dull')),
    nasolabialScore: round(meanOf('sag') * 0.5 + meanOf('tex') * 0.2 + meanOf('pig') * 0.1),
    
    // NEW: surface mesh availability flag (client can read rawViews for points)
    mesh468Enabled: FACE_MESH_ENABLE && MESH_READY ? true : false,
  };
  
  const suggestionIds = mapToTreatmentIds(findings);
  
  // Compact mesh into raw: only pass perView name + points
  const rawViews = {};
  for (const k of Object.keys(perView)) {
    const v = perView[k];
    rawViews[k] = {
      pose: v.pose,
      nearFrontal: v.nearFrontal,
      mesh468: v.mesh468 ? { count: v.mesh468.count, points: v.mesh468.points } : null,
    };
  }
  
  return { findings: { ...findings, views: rawViews }, suggestionIds };
}

// ============================================================================
// 11) Auth & Profile routes + middleware
// ============================================================================
function safeUser(u) {
  const { password, token, ...safe } = u;
  return safe;
}

function requireAuth(req, res, next) {
  const auth = req.headers['authorization'] || '';
  const m = auth.match(/^Bearer\s+(.+)$/i);
  if (!m) return res.status(401).json({ error: 'UNAUTHORIZED' });
  const token = m[1];
  const users = readJSON(USERS_DB, []);
  const user = users.find(u => u.token === token);
  if (!user) return res.status(401).json({ error: 'UNAUTHORIZED' });
  req.user = user;
  next();
}

app.post('/signup', (req, res) => {
  const { email, password } = req.body || {};
  if (!email || !password) return res.status(400).json({ error: 'MISSING_FIELDS', message: 'Please provide both email and password.' });
  
  const users = readJSON(USERS_DB, []);
  if (users.find(u => u.email.toLowerCase() === String(email).toLowerCase())) {
    return res.status(409).json({ error: 'EMAIL_EXISTS', message: 'This email is already registered.' });
  }
  
  const user = {
    id: 'u' + Date.now(),
    email: String(email).trim(),
    password: String(password),
    name: '',
    phone: '',
    token: 't' + crypto.randomBytes(16).toString('hex'),
  };
  users.push(user);
  writeJSON(USERS_DB, users);
  return res.json({ token: user.token, user: safeUser(user) });
});

app.post('/login', (req, res) => {
  const { email, password } = req.body || {};
  if (!email || !password) return res.status(400).json({ error: 'MISSING_FIELDS', message: 'Please provide both email and password.' });
  
  const users = readJSON(USERS_DB, []);
  const user = users.find(u => u.email.toLowerCase() === String(email).toLowerCase());
  if (!user || user.password !== String(password)) {
    return res.status(401).json({ error: 'INVALID_CREDENTIALS', message: 'Invalid email or password.' });
  }
  
  user.token = 't' + crypto.randomBytes(16).toString('hex');
  writeJSON(USERS_DB, users);
  return res.json({ token: user.token, user: safeUser(user) });
});

app.get('/profile/:id', requireAuth, (req, res) => {
  if (req.user.id !== req.params.id) return res.status(403).json({ error: 'FORBIDDEN' });
  return res.json(safeUser(req.user));
});

app.put('/profile/:id', requireAuth, (req, res) => {
  if (req.user.id !== req.params.id) return res.status(403).json({ error: 'FORBIDDEN' });
  const { name, phone } = req.body || {};
  const users = readJSON(USERS_DB, []);
  const idx = users.findIndex(u => u.id === req.user.id);
  if (idx === -1) return res.status(404).json({ error: 'NOT_FOUND' });
  if (typeof name === 'string') users[idx].name = name.trim();
  if (typeof phone === 'string') users[idx].phone = phone.trim();
  writeJSON(USERS_DB, users);
  return res.json(safeUser(users[idx]));
});

// ============================================================================
// 12) Analysis + status routes
// ============================================================================

app.get('/health', (_req, res) => res.send('OK'));

app.get('/ping', (_req, res) => res.json({ ok: true, backend: 'ready' }));

app.get('/status', (_req, res) => {
  res.set('Cache-Control', 'no-store');
  res.json({
    tfBackend: tf.getBackend(),
    modelsReady: MODELS_READY,
    modelSource: MODEL_SOURCE,
    modelUrl: MODEL_URL_USED,
    lastModelError: LAST_MODEL_ERROR,
    // NEW:
    meshEnabled: FACE_MESH_ENABLE,
    meshReady: MESH_READY,
    meshModel: MESH_MODEL_NAME || null,
  });
});

app.get('/preload', async (_req, res) => {
  try {
    await loadModelsOnce();
    await loadFaceMeshOnce(); // NEW: prepare mesh too (non-fatal if fails)
    return res.json({
      ok: true,
      modelsReady: true,
      modelSource: MODEL_SOURCE,
      modelUrl: MODEL_URL_USED,
      meshEnabled: FACE_MESH_ENABLE,
      meshReady: MESH_READY,
      meshModel: MESH_MODEL_NAME || null,
    });
  } catch (e) {
    console.error('Preload error:', e);
    return res.status(500).json({ ok: false, error: String(e) });
  }
});

// ==== /uploads (Expo-friendly, no timeouts, clear errors, JPEG normalize) ====
// ==== /uploads (no disk writes; in-memory storage with TTL) ====
app.post(
  '/uploads',
  requireAuth,
  (req, res, next) => { req.setTimeout(0); res.setTimeout(0); next(); },
  (req, res, next) => {
    upload.fields([
      { name: 'front', maxCount: 1 },
      { name: 'left', maxCount: 1 },
      { name: 'right', maxCount: 1 },
    ])(req, res, (err) => {
      if (err) {
        if (err instanceof multer.MulterError) {
          if (err.code === 'LIMIT_FILE_SIZE') {
            return res.status(413).json({ error: 'FILE_TOO_LARGE', detail: 'Max 20MB per image' });
          }
          return res.status(400).json({ error: 'UPLOAD_ERROR', detail: err.message });
        }
        return res.status(400).json({ error: 'UPLOAD_ERROR', detail: String(err?.message || err) });
      }
      next();
    });
  },
  async (req, res) => {
    try {
      const need = ['front','left','right'];
      const gotAll = need.every((k) => (req.files?.[k] || []).length === 1);
      if (!gotAll) {
        return res.status(400).json({ error: 'MISSING_IMAGES', detail: 'Need front, left, right' });
      }

      const uploadId = 'u' + Date.now() + '-' + Math.random().toString(36).slice(2);

      // Normalize & store buffers in memory with TTL; never touch disk
      for (const field of need) {
        const f = (req.files?.[field] || [])[0];
        const mime = String(f.mimetype || '').toLowerCase();
        if (!mime.startsWith('image/')) {
          return res.status(415).json({ error: 'ONLY_IMAGES_ALLOWED' });
        }
        const normalized = await ensureJpegBuffer(f.buffer);
        const key = `mem:${uploadId}:${field}`;
        putMem(key, normalized, mime);
      }

      const files = need.map((field) => ({ field, path: `mem:${uploadId}:${field}` }));

      console.log('[uploads] ok (in-memory)', files.map(f => `${f.field}=${f.path}`).join(' '), '| user', req.user.id);
      res.json({ uploadId, files });
    } catch (err) {
      console.error('[uploads] failed:', err?.stack || err);
      res.status(500).json({ error: 'UPLOAD_FAILED' });
    }
  }
);


const DISCLAIMER = 'The information provided is for educational purposes only and is not medical advice or a diagnosis. Personal recommendations require in-person assessment by a qualified clinician.';

app.post('/analysis/start', requireAuth, (req, res, next) => {
  res.setTimeout(0);
  next();
}, async (req, res) => {
  const t0 = Date.now();

  // helper: turn request "files" values into Buffers (mem:...) or legacy file paths
  function resolveInput(any) {
    const s = String(any || '');
    if (s.startsWith('mem:')) {
      const buf = getMem(s);         // <-- from the in-memory store you added
      if (!buf) throw new Error('MISSING_IN_MEMORY_IMAGE');
      return buf;                    // Buffer is accepted by loadImage(...)
    }
    // Backward-compat: if a plain filename/path was posted, allow it
    return path.join(UPLOAD_DIR, path.basename(s));
  }

  // helper: for report.files — show mem keys as-is; legacy as basename
  function displayFileId(any) {
    const s = String(any || '');
    return s.startsWith('mem:') ? s : path.basename(s);
  }

  try {
    let files = req.body?.files;
    if (typeof files === 'string') {
      try { files = JSON.parse(files); } catch {}
    }
    if (!files?.front || !files?.left || !files?.right) {
      return res.status(400).json({ error: 'MISSING_IMAGES' });
    }

    // Build inputs that can be Buffers (for mem: keys) or paths (legacy)
    const inputs = {
      front: resolveInput(files.front),
      left:  resolveInput(files.left),
      right: resolveInput(files.right),
    };

    // Run analysis (your analyzeThree now loads from Buffer OR path)
    const { findings, suggestionIds } = await analyzeThree(inputs);

    // If inputs came from memory keys, purge them now
    for (const k of ['front', 'left', 'right']) {
      const s = String(files[k] || '');
      if (s.startsWith('mem:')) delMem(s);
    }

    // Build suggestions
    const treatments = loadTreatments().items;
    const suggestionCards = suggestionIds.map(id => {
      const t = treatments.find(tt => tt.id === id);
      return t ? {
        id: t.id,
        name: t.name,
        description: t.description,
        durationMin: t.durationMin,
        priceMin: t.priceMin,
        priceMax: t.priceMax,
      } : null;
    }).filter(Boolean);

    // Narrative
    const aiReportEnglish = buildEnglishNarrative(findings, suggestionCards, DISCLAIMER);

    // Top details
    const detailsTop = [
      ['Forehead lines', findings.foreheadWrinkleScore],
      ['Glabellar (frown) lines', findings.glabellarScore],
      ['Crow’s feet', findings.crowsFeetScore],
      ['Under-eye darkness', findings.darkCircleScore],
      ['Eye-bag puffiness', findings.eyebagScore],
      ['Temple hollowing', findings.templeHollownessScore],
      ['Sagging/laxity', findings.saggingScore],
      ['Nasolabial folds', findings.nasolabialScore],
      ['Perioral fine lines', findings.perioralFineLinesScore],
      ['Thin lips', findings.thinLipsScore],
      ['Marionette lines', findings.marionetteScore],
      ['Weak chin', findings.weakChinScore],
      ['Jawline softness', findings.jawlineSoftnessScore],
      ['Double-chin fullness', findings.doubleChinScore],
      ['Redness', findings.rednessScore],
      ['Pigmentation', findings.pigmentationScore],
      ['Texture/pores/scarring', findings.textureScore],
      ['Dullness', findings.dullSkinScore],
    ].sort((a,b)=>b[1]-a[1]).filter(([,v])=>v>=25).slice(0,5).map(([k])=>k);

    // Summary strings
    const ageSummary = Number.isFinite(findings.ageEstimate)
      ? `Age estimate: ~${findings.ageEstimate} (${findings.ageLow}–${findings.ageHigh}).`
      : 'Age estimate: —.';
    const symText = Number.isFinite(findings.symRmsMm) ? `${findings.symRmsMm.toFixed(1)} mm` : '-';
    const symPctText = Number.isFinite(findings.symPctIPD) ? `(${(findings.symPctIPD*100).toFixed(1)}% IPD)` : '';

    // Final report
    const report = {
      id: 'r' + Date.now(),
      userId: req.user.id,
      createdAt: new Date().toISOString(),
      summary: `Face asymmetry: ${symText} ${symPctText}, ${findings.symBucket}. ${ageSummary}`,
      metrics: {
        emotion: findings.topEmotion || '-',
        symmetry: findings.symmetry,
        symmetryMm: findings.symRmsMm,
        symmetryPctIPD: findings.symPctIPD,
        symmetryBucket: findings.symBucket,
        symmetryStdMm: findings.symStdMm,
        glasses: false,
        ageEstimate: findings.ageEstimate,
        ageLow: findings.ageLow,
        ageHigh: findings.ageHigh,
      },
      details: detailsTop,
      suggestions: suggestionCards,
      disclaimer: DISCLAIMER,
      aiReportGreek: aiReportEnglish, // frontend still reads this key
      files: {
        // IMPORTANT: show mem keys as-is; legacy filenames as basename
        front: displayFileId(files.front),
        left:  displayFileId(files.left),
        right: displayFileId(files.right),
      },
      raw: findings,
      perfMs: Date.now() - t0,
    };

    // Persist report
    const all = readJSON(ANALYSES_DB, []);
    all.unshift(report);
    writeJSON(ANALYSES_DB, all);

    console.log(
      '[analysis] ok in', report.perfMs, 'ms',
      '| user:', req.user.id,
      '| sym:', Number.isFinite(findings.symRmsMm) ? findings.symRmsMm.toFixed(2)+'mm' : '-',
      '| age:', Number.isFinite(findings.ageEstimate) ? `${findings.ageEstimate} (${findings.ageLow}-${findings.ageHigh})` : '—',
      '| mesh:', findings.mesh468Enabled ? 'on' : 'off'
    );

    res.json(report);
  } catch (err) {
    console.error('analysis/start FAILED:', err?.stack || err);
    const msg = String(err?.message || err);
    if (msg === 'NO_FACE_DETECTED') return res.status(422).json({ error: 'NO_FACE_DETECTED' });
    if (msg.startsWith('MISSING_IMAGE_')) return res.status(400).json({ error: msg });
    if (msg === 'ONLY_IMAGES_ALLOWED') return res.status(415).json({ error: 'ONLY_IMAGES_ALLOWED' });
    if (msg === 'MISSING_IN_MEMORY_IMAGE') return res.status(400).json({ error: msg });
    res.status(500).json({ error: 'ANALYSIS_FAILED', detail: msg });
  }
});


// Per-user history & report access
app.get('/history', requireAuth, (req, res) => {
  const all = readJSON(ANALYSES_DB, []);
  const mine = all.filter(r => r.userId === req.user.id);
  res.json(mine.map(r => ({ id: r.id, createdAt: r.createdAt, summary: r.summary })));
});

app.get('/reports/:id', requireAuth, (req, res) => {
  const all = readJSON(ANALYSES_DB, []);
  const r = all.find(x => x.id === req.params.id);
  if (!r) return res.status(404).json({ error: 'NOT_FOUND' });
  if (r.userId !== req.user.id) return res.status(403).json({ error: 'FORBIDDEN' });
  res.json(r);
});

app.get('/treatments', (_req, res) => {
  res.json(loadTreatments());
});

// ============================================================================
// 13) Start server, preload models, print LAN URLs
// ============================================================================
function lanIPv4s() {
  const nets = os.networkInterfaces();
  const out = [];
  for (const name of Object.keys(nets)) {
    for (const n of nets[name] || []) {
      if (n.family === 'IPv4' && !n.internal) out.push(n.address);
    }
  }
  return out;
}

(async () => {
  try {
    await loadModelsOnce();
    await loadFaceMeshOnce(); // NEW: attempt mesh preload
    console.log('Models preloaded at startup.');
  } catch (e) {
    console.error('Model preload failed:', e);
  }
})();

app.listen(PORT, '0.0.0.0', () => {
  const lans = lanIPv4s();
  console.log('=== AESTHETIC-AI BACKEND (auth+analysis) ===');
  console.log('API listening on:');
  console.log(` • Local: http://localhost:${PORT}`);
  lans.forEach(ip => console.log(` • LAN: http://${ip}:${PORT}`));
  console.log(`Models dir: ${MODELS_DIR}`);
  console.log(`Uploads dir: ${UPLOAD_DIR}`);
});