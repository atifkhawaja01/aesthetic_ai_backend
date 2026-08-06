const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'auth-regression-'));
const userDb = path.join(tmpDir, 'users.json');
fs.writeFileSync(userDb, '[]');

const dataDir = tmpDir;
process.env.DATA_DIR = dataDir;
process.env.USERS_DB_PATH = userDb;

const serverModulePath = path.join(__dirname, '..', 'index.js');
const { app } = require(serverModulePath);

(async () => {
  const port = 4101;
  const server = app.listen(port);
  await new Promise((resolve) => server.once('listening', resolve));

  try {
    const signupRes = await fetch(`http://127.0.0.1:${port}/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: 'regression@example.com', password: 'pass1234' }),
    });

    assert.equal(signupRes.status, 200, 'signup should succeed');
    const signupBody = await signupRes.json();
    assert.ok(signupBody.token, 'signup should return a token');

    const loginRes = await fetch(`http://127.0.0.1:${port}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: 'regression@example.com', password: 'pass1234' }),
    });

    assert.equal(loginRes.status, 200, 'login should succeed');
    const loginBody = await loginRes.json();
    assert.ok(loginBody.token, 'login should return a token');
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
})();
