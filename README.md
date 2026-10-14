# AI Troubleshooter (Agent 2)

A Flask + LangGraph + Google Gemini troubleshooting assistant with a local rule-based fallback for common problems.

## Run locally (Windows)

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

5. Copy `.env.example` to `.env` and set `GOOGLE_API_KEY` to your own Gemini API key. The local fallback can still operate without a key, but Gemini-generated responses will be unavailable.
6. Start the app:

   ```powershell
   python app.py
   ```

7. Open `http://127.0.0.1:5000` and verify `http://127.0.0.1:5000/health` returns JSON.

## Render deployment

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn app:app`
- **Environment Variables:** `GOOGLE_API_KEY` (secret), `GEMINI_MODEL` = `gemini-2.5-flash`

Render provides the `PORT` environment variable. Gunicorn binds to the Render web-service port automatically.

Never commit `.env` or put an API key in frontend JavaScript, screenshots, or public repository files.

## API

- `GET /` — web interface
- `GET /health` — health status and non-secret configuration metadata
- `POST /api/message` — JSON body: `{"messages":[{"role":"user","content":"My Wi-Fi works on my phone but not my laptop"}]}`

The health endpoint reports application health; it does not guarantee that Gemini credentials or external API connectivity are valid.
