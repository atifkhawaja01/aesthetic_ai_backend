// scripts/fetch-models.js
const fs = require('fs');
const path = require('path');
const https = require('https');

const MODELS_DIR = path.join(__dirname, '..', 'models');
// use a pinned version path; change if you update package version later:
const BASE = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model';

const manifests = [
  'ssd_mobilenetv1_model-weights_manifest.json',
  'face_landmark_68_model-weights_manifest.json',
  'age_gender_model-weights_manifest.json',
  'face_expression_model-weights_manifest.json',
];

function get(url) {
  return new Promise((resolve, reject) => {
    https.get(url, (r) => {
      if (r.statusCode !== 200) return reject(new Error(`HTTP ${r.statusCode} for ${url}`));
      const chunks = [];
      r.on('data', (d) => chunks.push(d));
      r.on('end', () => resolve(Buffer.concat(chunks)));
    }).on('error', reject);
  });
}

async function ensureDir(p) {
  await fs.promises.mkdir(p, { recursive: true });
}

async function downloadFile(rel) {
  const url = `${BASE}/${rel}`;
  const dest = path.join(MODELS_DIR, rel);
  await ensureDir(path.dirname(dest));
  const buf = await get(url);
  await fs.promises.writeFile(dest, buf);
  console.log('✓', rel);
  return dest;
}

async function main() {
  await ensureDir(MODELS_DIR);
  let total = 0;

  for (const mf of manifests) {
    // 1) download manifest
    const manifestPath = await downloadFile(mf);
    total++;

    // 2) read manifest and download all shard paths
    const manifest = JSON.parse(await fs.promises.readFile(manifestPath, 'utf8'));
    const groups = manifest.weights || manifest.manifest; // supports tfjs styles
    if (!groups) continue;

    for (const g of groups) {
      const paths = g.paths || [];
      for (const p of paths) {
        await downloadFile(p);
        total++;
      }
    }
  }

  console.log(`\nAll done. Downloaded ${total} files into ${MODELS_DIR}`);
}

main().catch((e) => {
  console.error('Download failed:', e.message);
  process.exit(1);
});
