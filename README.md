# Sclera Biometrics

Run order for local development:
1. Start the backend first
2. Start the frontend second

These instructions are for this repository layout:
- Backend app: `Backend/FastAPI.py`
- Frontend app: `Frontend/`

## Prerequisites

- Python 3.10+
- Node.js 18+

## 1. Start Backend (First)

Open Terminal 1 from the repository root.

```bash
cd Backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn FastAPI:app --host 0.0.0.0 --port 8000 --reload
```

Backend should be available at:
- http://127.0.0.1:8000

Quick health check (optional):

```bash
curl http://127.0.0.1:8000/liveness_check
```

Expected response:

```json
{"ok": true}
```

## 2. Start Frontend (Second)

Open Terminal 2 from the repository root.

```bash
cd Frontend
npm install
npm run dev
```

Open the Vite URL shown in terminal (usually http://127.0.0.1:5173).

## 3. Optional: Run Electron Shell

From `Frontend/`:

```bash
npm run start
```

This runs Vite and Electron together.

## Notes

- Keep both backend and frontend terminals running while testing.
- Backend CORS already allows common local frontend ports including `5173`.
