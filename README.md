# ✨ HuntMate AI (Streamlit)

An autonomous AI job-application agent, originally built as a Google Colab
notebook with a Gradio UI, converted here to a **Streamlit** app so it can be
deployed on Streamlit Community Cloud, Hugging Face Spaces, or any other host
that runs a `streamlit run app.py` app.

## What it does

1. Reads your resume (PDF) and extracts the text.
2. Sends a master prompt to Gemini (`gemini-3.1-flash-lite`) describing the
   candidate and the desired job/location.
3. When Gemini asks to search, the app performs a **real** DuckDuckGo search
   (no fake/hallucinated job data allowed by the prompt).
4. Gemini selects one real job posting from the search results and drafts a
   tailored application email using only facts from your resume.
5. If a verified recipient email is found, the app sends the application via
   Gmail SMTP. Otherwise, it prepares the email but does not send it.

## Files

```
resume-agent/
├── app.py                          # Streamlit app (main entry point)
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── secrets.toml.example        # Template for local secrets (copy, fill in, rename)
├── .gitignore
└── README.md
```

## Setup — Run locally

```bash
git clone <your-repo-url>
cd resume-agent
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure secrets
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and add your real keys

streamlit run app.py
```

## Required secrets / API keys

| Key | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Google Gemini API key ([Google AI Studio](https://aistudio.google.com/apikey)) |
| `GMAIL_ADDRESS` | Optional | Gmail address used to actually send the application email |
| `GMAIL_APP_PASSWORD` | Optional | Gmail **App Password** (not your normal password) — requires 2-Step Verification enabled on the Google account |

Without the Gmail secrets, the agent still works end-to-end — it just won't
send the final email; it will present it as "Application Prepared - Not Sent".

## Deploy on Streamlit Community Cloud

1. Push this folder to a **public or private GitHub repository**.
2. Go to https://share.streamlit.io and click **"New app"**.
3. Select your repo, branch, and set the main file path to `app.py`.
4. Before (or after) deploying, open **App settings → Secrets** and paste:

   ```toml
   GEMINI_API_KEY = "your-gemini-api-key"
   GMAIL_ADDRESS = "youraddress@gmail.com"
   GMAIL_APP_PASSWORD = "your-app-password"
   TAVILY_API_KEY = "your-tavily-api-key"

   ```

5. Click **Deploy**. Streamlit Cloud installs `requirements.txt` automatically.

## Notes on the conversion from Gradio/Colab

- `google.colab.userdata.get(...)` → replaced with `st.secrets` (falls back to
  `os.environ` for local/non-Streamlit use).
- `gr.File(type="filepath")` → `st.file_uploader(...)`, and `PdfReader` now
  reads directly from the uploaded file object (in-memory), since Streamlit
  Cloud has no persistent Colab-style filesystem to point a path at.
- `gr.Textbox` (output/logs) → a live-updating `st.text_area` fed by a small
  `log()` helper, plus a final Markdown-rendered report.
- The custom Gradio CSS/theme was reworked into a lighter Streamlit-native
  layout (columns + `st.markdown` HTML snippets) — same information panels,
  Streamlit look and feel.
- All core agent logic — `search_jobs`, `send_application_email`,
  `run_autonomous_agent`, and the master prompt — is unchanged from the
  original notebook.

## Disclaimer

This agent can autonomously send real emails on your behalf once Gmail
secrets are configured. Review the generated application in the log output
before relying on it, and test with a personal address first.
