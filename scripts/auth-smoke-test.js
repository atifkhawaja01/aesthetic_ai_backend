const baseUrl = process.env.API_BASE || 'http://localhost:4000';

async function postJson(path, body) {
  const res = await fetch(`${baseUrl}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status} ${res.statusText}: ${text}`);
  }

  return res.json();
}

async function run() {
  const email = `smoke_${Date.now()}@example.com`;
  const password = 'pass1234';

  const signup = await postJson('/signup', { email, password });
  if (!signup?.token) throw new Error('Signup did not return a token');

  const firstLogin = await postJson('/login', { email, password });
  if (!firstLogin?.token) throw new Error('Login did not return a token');

  const secondLogin = await postJson('/login', { email, password });
  if (!secondLogin?.token) throw new Error('Second login did not return a token');

  console.log('Auth smoke test passed:', { email });
}

run().catch((err) => {
  console.error('Auth smoke test failed:', err?.message || err);
  process.exit(1);
});
