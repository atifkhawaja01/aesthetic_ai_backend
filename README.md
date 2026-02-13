# Lakidis Aesthetic — Backend (Node/Express)

API with **no external integrations** (demo mode). Stores uploads on disk and analysis/history in JSON files.

## Quick Start
```bash
cd backend
npm i
npm run start
```

The API listens on **:4000** by default.

## Routes
- POST `/signup` `{ email, password }`
- POST `/login` `{ email, password }`
- POST `/auth/register` `{ email, password }` (alias)
- POST `/auth/login` `{ email, password }` (alias)
- POST `/uploads` form-data with `front`, `left`, `right`
- POST `/analysis/start` `{ uploadId }` → returns demo analysis object
- GET `/history` → list of past demo summaries
- GET `/treatments` → seeded from `seed/treatments.seed.json`
- GET `/config` → site url + disclaimer

## Notes
- Replace the logic inside `/analysis/start` later to call your real AI pipelines (Azure Face, OpenAI, etc.).
- Add authentication and database as needed (JWT, Postgres, etc.).

## Persistence (Render)
- User data is stored in a JSON file. By default it uses `backend/data/users.json`.
- For production, set `DATA_DIR` (or `USERS_DB_PATH`) to a mounted persistent disk path.
	- Example: create a Render disk mounted at `/var/data` and set `DATA_DIR=/var/data`.
- Ensure your Render service is configured with the correct Vercel origin in CORS.

## Persistence (Supabase - Free)
If you do not want a paid disk, you can store users in Supabase Postgres.

1) Create a `users` table in Supabase (SQL editor):
```sql
create table if not exists public.users (
	id text primary key,
	email text unique not null,
	password_hash text not null,
	name text,
	phone text,
	token text,
	created_at timestamp with time zone default now()
);
```

2) Set these env vars on Render (backend only):
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`

The backend will automatically use Supabase when these are present.
# aesthetic_ai_backend
