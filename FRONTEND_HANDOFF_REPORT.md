# Frontend Handoff Report — Auth, History, Upload, and Analysis Flow

## Purpose
This report is intended for a frontend repository agent that needs to update the client app so it works correctly with the backend changes made for auth, history, image upload, and analysis.

The backend is now more production-safe, but the frontend must align with the new contract. The most important areas are:
- authentication/session handling
- history page behavior
- upload -> analyze workflow
- error handling and loading states
- request headers and payload shapes

---

## 1. Summary of Backend Changes
The backend now expects and supports the following behavior:

### Authentication
- Login and signup return:
  ```json
  {
    "token": "...",
    "user": {
      "id": "...",
      "email": "...",
      "name": "",
      "phone": ""
    }
  }
  ```
- Protected endpoints require a token.
- Token can be provided in the following forms:
  - `Authorization: Bearer <token>`
  - `x-auth-token: <token>`
  - `x-token: <token>`
  - `token: <token>`
  - query param `?token=<token>`

### Upload flow
The upload endpoint now returns:
```json
{
  "uploadId": "u123",
  "files": [
    { "field": "front", "path": "mem:u123:front" },
    { "field": "left", "path": "mem:u123:left" },
    { "field": "right", "path": "mem:u123:right" }
  ],
  "fileMap": {
    "front": "mem:u123:front",
    "left": "mem:u123:left",
    "right": "mem:u123:right"
  }
}
```

### Analysis flow
The analysis endpoint accepts any of the following payload styles:
```json
{ "uploadId": "u123" }
```
or
```json
{
  "files": {
    "front": "mem:u123:front",
    "left": "mem:u123:left",
    "right": "mem:u123:right"
  }
}
```
or
```json
{
  "fileMap": {
    "front": "mem:u123:front",
    "left": "mem:u123:left",
    "right": "mem:u123:right"
  }
}
```

### History and reports
- `GET /history` returns a list of summaries for the logged-in user.
- `GET /reports/:id` returns the full report for a specific analysis.

---

## 2. Demo Auth Mode (required for this deployment)
The backend is now configured for a demo-only auth experience. The frontend should not depend on signup persistence or a real user database.

### Required behavior
- The login screen should be prefilled with a demo credential.
- The frontend should call either `/login` or `/signup` with the same demo email/password and expect a success response with a token and user object.
- The frontend should not show a signup failure when the backend is in demo mode.
- The frontend should store the returned token locally and use it for protected requests.

### Backend contract
- `POST /login` accepts any non-empty email/password and returns:
  ```json
  {
    "token": "...",
    "user": {
      "id": "demo-user",
      "email": "demo@example.com",
      "name": "Demo User",
      "phone": ""
    }
  }
  ```
- `POST /signup` behaves the same way.
- `GET /auth/demo-credentials` returns the default demo values:
  ```json
  {
    "email": "demo@example.com",
    "password": "demo1234"
  }
  ```

### Frontend implementation notes
- Prefer reading the demo credentials from `/auth/demo-credentials` on first load.
- If that endpoint is unavailable, fall back to the hardcoded values `demo@example.com` and `demo1234`.
- After login, navigate to the main app and persist the token in `localStorage`.

---

## 3. Frontend Impact Areas

### A. Authentication state management
The frontend must be updated to ensure the user does not get logged out unexpectedly after navigating to history or profile pages.

#### Required changes
1. Store the login/signup token in persistent client storage (preferably `localStorage` or `sessionStorage`).
2. On app startup, restore the token from storage and attach it to future requests.
3. Do not clear the session just because the history page or profile page request fails once due to missing auth headers.
4. Ensure the app uses a single shared auth helper for all protected requests.

#### Expected frontend behavior
- After login, the app should remain authenticated across page refreshes.
- The history page should load only when the token is attached.
- If the token is missing or invalid, the app should redirect to login rather than silently breaking.

#### Recommended implementation pattern
Use a shared helper such as:
```ts
function getAuthHeaders() {
  const token = localStorage.getItem('authToken');
  return token ? { Authorization: `Bearer ${token}` } : {};
}
```

---

### B. Login/signup flow
The frontend should expect the backend response shape exactly as returned by the API.

#### Backend response contract
```json
{
  "token": "...",
  "user": {
    "id": "...",
    "email": "...",
    "name": "",
    "phone": ""
  }
}
```

#### Frontend actions
- On successful login/signup:
  - save `token`
  - save `user`
  - navigate to the next screen
- On failure:
  - show a friendly message
  - do not leave the user in a broken auth state

---

### C. History page
The history page currently depends on auth. It must use the auth token consistently.

#### Required behavior
- Send `Authorization: Bearer <token>` on `GET /history`.
- Handle empty history gracefully.
- When clicking a history item, call `GET /reports/:id` with the same auth headers.
- If the backend responds with `401`, redirect to login and clear stale auth state.

#### Expected response shape
```json
[
  {
    "id": "r123",
    "createdAt": "2026-08-06T00:00:00.000Z",
    "summary": "..."
  }
]
```

#### Frontend update
- Make the history screen resilient to missing data.
- If a report fails to load, show a clear fallback message instead of logging out the user.

---

### D. Upload and analysis workflow
This is the most important client-side integration area.

#### What the frontend currently needs to do
1. Upload images using multipart form-data with fields:
   - `front`
   - `left`
   - `right`
2. Send the auth token in the request headers.
3. Read the upload response:
   - `uploadId`
   - `files`
   - `fileMap`
4. Call analysis using the returned `uploadId`.

#### Strongly recommended frontend change
After upload succeeds, immediately start analysis with:
```json
{ "uploadId": "<returned-uploadId>" }
```

This is the simplest and most reliable contract.

#### Backend response after analysis success
The frontend should be ready to read:
```json
{
  "id": "r123",
  "userId": "u123",
  "createdAt": "...",
  "summary": "...",
  "metrics": {
    "emotion": "...",
    "symmetry": 0,
    "symmetryMm": 0,
    "symmetryPctIPD": 0,
    "symmetryBucket": "...",
    "symmetryStdMm": 0,
    "glasses": false,
    "ageEstimate": 0,
    "ageLow": 0,
    "ageHigh": 0
  },
  "details": ["..."],
  "suggestions": [],
  "disclaimer": "...",
  "aiReportGreek": "...",
  "files": {
    "front": "mem:u123:front",
    "left": "mem:u123:left",
    "right": "mem:u123:right"
  },
  "raw": {},
  "perfMs": 1234
}
```

#### Important frontend note
If the model pipeline is unavailable or the image is invalid, the backend may return errors such as:
- `BAD_IMAGE_INPUT`
- `NO_FACE_DETECTED`
- `MISSING_IMAGES`
- `ANALYSIS_FAILED`

The frontend should show helpful messages and not crash the UI.

---

### E. Error handling and UI resilience
The frontend should not assume every request will succeed.

#### Errors to handle explicitly
- `401 UNAUTHORIZED` → redirect to login
- `403 FORBIDDEN` → show access denied message
- `400 MISSING_FIELDS` → show validation message
- `409 EMAIL_EXISTS` → show account already exists message
- `415 ONLY_IMAGES_ALLOWED` / `BAD_IMAGE_INPUT` → show image format issue message
- `422 NO_FACE_DETECTED` → show facial detection failure message
- `500 AUTH_STORAGE_ERROR` / `ANALYSIS_FAILED` → show generic retry message

#### Frontend UX recommendation
Use a shared error mapper such as:
```ts
function getUserFriendlyError(message: string) {
  if (message === 'BAD_IMAGE_INPUT') return 'The selected images could not be processed. Please try again with a clearer image.';
  if (message === 'NO_FACE_DETECTED') return 'No face was detected. Please upload clearer photos.';
  if (message === 'MISSING_IMAGES') return 'Please upload front, left, and right images.';
  return 'Something went wrong. Please try again.';
}
```

---

## 3. Recommended Frontend Changes by File/Area

### Auth layer
Update the API client or auth service to:
- attach the token to protected requests
- persist the token after login/signup
- restore the token on app startup
- clear the token only when the user logs out

### Login/signup screens
- use the new response shape from the backend
- store `token` and `user`
- route to the main app after success

### History screen
- request `/history` with auth headers
- render entries from the response
- open a report detail screen using `/reports/:id`

### Upload screen
- send multipart form-data with `front`, `left`, `right`
- include the auth token
- capture and store the returned `uploadId`
- move to the analysis step immediately after upload success

### Analysis screen
- send `{ uploadId }` to `/analysis/start`
- handle success, no-face-detected, bad-image, and generic errors
- render the report fields from the backend response

---

## 4. Suggested Implementation Checklist

### Highest priority
- [ ] Persist auth token after login/signup
- [ ] Attach token to protected requests
- [ ] Fix history page auth flow
- [ ] Fix upload -> analyze payload handoff
- [ ] Add robust error handling for image and analysis failures

### Medium priority
- [ ] Add loading spinners and disabled states during upload/analyze
- [ ] Improve empty state for history page
- [ ] Display backend errors in a user-friendly way
- [ ] Preserve auth on refresh

### Nice to have
- [ ] Add retry support for transient analysis failures
- [ ] Show backend processing time or progress indication
- [ ] Keep the upload state if the user navigates away temporarily

---

## 5. Copy-Paste Prompt for the Frontend Repo Agent
Use the following prompt with the frontend repository agent:

```text
Please inspect the frontend app and update it to match the backend auth/upload/analyze contract that was just stabilized.

Context:
- The backend now returns a token on signup/login and expects protected requests to include it via Authorization: Bearer <token>.
- The frontend must persist the token and restore it on reload so users do not get logged out when opening history/profile pages.
- The upload endpoint returns { uploadId, files, fileMap } and the analysis endpoint should be called with { uploadId } after a successful upload.
- The frontend should handle error responses such as 401, 403, 400, 409, 415, 422, and 500 gracefully with friendly user-facing messages.
- The history page should call GET /history and GET /reports/:id using the same auth token.
- The upload flow should use multipart form-data with front/left/right image fields.
- The analysis view should render the report fields from the backend response, including summary, metrics, details, suggestions, disclaimer, and report text.

Please make the necessary frontend updates across auth, history, upload, analyze, and error handling.
Focus on production readiness and resilience rather than cosmetic changes.
```

---

## 6. Final Note
The biggest frontend risk is not the UI styling; it is the integration contract between auth, upload, and analysis. If the frontend uses the wrong request shape or forgets to send the auth token, the user experience will break in exactly the ways we just hardened on the backend.