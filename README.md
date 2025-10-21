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
- POST `/auth/register` `{ email, password }`
- POST `/auth/login` `{ email, password }`
- POST `/uploads` form-data with `front`, `left`, `right`
- POST `/analysis/start` `{ uploadId }` → returns demo analysis object
- GET `/history` → list of past demo summaries
- GET `/treatments` → seeded from `seed/treatments.seed.json`
- GET `/config` → site url + disclaimer

## Notes
- Replace the logic inside `/analysis/start` later to call your real AI pipelines (Azure Face, OpenAI, etc.).
- Add authentication and database as needed (JWT, Postgres, etc.).
# aesthetic_ai_backend
