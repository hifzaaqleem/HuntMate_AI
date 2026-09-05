"""
Gemini Career Agent — Autonomous Job Application Assistant
Streamlit version (converted from the original Gradio / Colab notebook)

Deploy on Streamlit Community Cloud:
1. Push this repo to GitHub.
2. On https://share.streamlit.io create a new app pointing at app.py.
3. In the app's "Secrets" settings, add:

    GEMINI_API_KEY = "your-gemini-api-key"
    GMAIL_ADDRESS = "youraddress@gmail.com"
    GMAIL_APP_PASSWORD = "your-16-char-app-password"

   (GMAIL_* are only required if you want the agent to actually send emails.
   Without them the agent will still search, match and draft the application,
   it just won't send it.)

Job search uses Gemini's native Google Search grounding tool rather than
scraping DuckDuckGo directly — this avoids cloud-host IP blocks and needs
no extra API key beyond GEMINI_API_KEY.
"""

import os
import re
import smtplib
import traceback
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import streamlit as st
from pypdf import PdfReader

from google import genai
from google.genai import types
from google.genai.types import HttpOptions, HttpRetryOptions

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Gemini Career Agent",
    page_icon="✨",
    layout="wide",
)

# ============================================================
# SECRETS / CONFIG HELPERS
# ============================================================


def get_secret(key: str) -> str | None:
    """Read a secret from Streamlit secrets first, then env vars."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key)


GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GMAIL_ADDRESS = get_secret("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = get_secret("GMAIL_APP_PASSWORD")

GEMINI_MODEL = "gemini-3.1-flash-lite"


@st.cache_resource(show_spinner=False)
def get_client(api_key: str):
    """Create (and cache) the Gemini client."""
    os.environ["GEMINI_API_KEY"] = api_key
    return genai.Client(
        http_options=HttpOptions(
            api_version="v1",
            retry_options=HttpRetryOptions(attempts=5, initial_delay=2.0),
        )
    )


# ============================================================
# 1. SEARCH JOBS (Gemini + native Google Search grounding)
# ============================================================
#
# NOTE ON THIS CHANGE:
# The original notebook scraped DuckDuckGo's HTML search page directly.
# That works fine from a Colab notebook, but most cloud hosts (Streamlit
# Community Cloud included) run on data-center IP ranges that DuckDuckGo
# aggressively blocks or CAPTCHAs — so the scrape silently returns zero
# results once deployed, even though nothing in the Python code is "broken".
#
# This version instead uses Gemini's built-in Google Search grounding tool
# (`types.Tool(google_search=types.GoogleSearch())`). The search runs on
# Google's own infrastructure, not from your app's server, so it isn't
# subject to that IP-blocking problem, and it needs no extra API key beyond
# the GEMINI_API_KEY you already have configured.


def search_jobs(query: str, client, log) -> str:
    try:
        log(f"🔎 Searching for jobs (Google Search grounding): {query}")

        grounding_tool = types.Tool(google_search=types.GoogleSearch())

        search_prompt = f"""
Search for REAL, currently open job postings matching:
{query}

Return up to 8 distinct postings. For EACH one, output in this exact format:

JOB RESULT
TITLE: <exact job title>
COMPANY: <company name, or "Unknown" if not visible>
LOCATION: <location>
URL: <direct URL to the posting>

Only include postings you can verify from real search results.
Do not invent any job, company, location, or URL.
If you find no real postings, respond with exactly: NO_RESULTS
"""

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=search_prompt,
            config=types.GenerateContentConfig(
                tools=[grounding_tool],
                temperature=0.1,
            ),
        )

        result_text = (response.text or "").strip()

        if not result_text or "NO_RESULTS" in result_text:
            log("⚠️ No job results found.")
            return (
                "NO JOB SEARCH RESULTS FOUND.\n\n"
                "No suitable real job posting was returned.\n\n"
                "DO NOT INVENT:\n- Job\n- Company\n- URL\n- Email\n"
            )

        result_count = result_text.count("JOB RESULT")
        log(f"✅ Found {max(result_count, 1)} job result(s).")
        return result_text

    except Exception as e:
        log(f"❌ Job search error: {str(e)}")
        return f"JOB SEARCH FAILED\n\nError:\n{str(e)}\n\nDO NOT INVENT JOB INFORMATION.\n"


# ============================================================
# 2. SEND APPLICATION EMAIL (Gmail SMTP)
# ============================================================


def send_application_email(recipient_email, job_title, company, candidate_name, email_body, log):
    try:
        log("📧 Preparing application email...")

        if not recipient_email:
            return (
                "## ⚠️ APPLICATION NOT SENT\n\n"
                "No verified recipient email address was found.\n\n"
                "The agent will NOT invent an email address."
            )

        email_pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
        if not re.match(email_pattern, recipient_email):
            return (
                f"## ⚠️ APPLICATION NOT SENT\n\n"
                f"The discovered recipient email does not appear valid:\n\n`{recipient_email}`"
            )

        if not GMAIL_ADDRESS:
            return (
                "## ❌ EMAIL CONFIGURATION ERROR\n\n"
                "Missing secret: `GMAIL_ADDRESS`\n\n"
                "Add it in Streamlit Cloud → App settings → Secrets."
            )

        if not GMAIL_APP_PASSWORD:
            return (
                "## ❌ EMAIL CONFIGURATION ERROR\n\n"
                "Missing secret: `GMAIL_APP_PASSWORD`\n\n"
                "Create a Google App Password and add it to Streamlit secrets."
            )

        subject = f"Application for {job_title}"
        message = MIMEMultipart()
        message["From"] = GMAIL_ADDRESS
        message["To"] = recipient_email
        message["Subject"] = subject
        message.attach(MIMEText(email_body, "plain", "utf-8"))

        log(f"📨 Sending to: {recipient_email}")

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, recipient_email, message.as_string())

        log("✅ Email sent successfully.")
        return (
            f"## ✅ APPLICATION SENT\n\n"
            f"**Recipient:** {recipient_email}\n\n"
            f"**Job:** {job_title}\n\n"
            f"**Company:** {company}\n\n"
            f"**Status:** Successfully sent"
        )

    except smtplib.SMTPAuthenticationError:
        return (
            "## ❌ GMAIL AUTHENTICATION FAILED\n\n"
            "Gmail rejected the login. Check that:\n\n"
            "1. 2-Step Verification is enabled.\n"
            "2. You created a Google App Password.\n"
            "3. The secret is named exactly `GMAIL_APP_PASSWORD`.\n"
            "4. You are using the App Password, NOT your normal Gmail password."
        )
    except Exception as e:
        log(f"❌ Email error: {str(e)}")
        return f"## ❌ EMAIL ERROR\n\n```text\n{str(e)}\n```"


# ============================================================
# 3. RUN AUTONOMOUS AGENT
# ============================================================


def run_autonomous_agent(resume_bytes, job_title, location, client, log):
    if not resume_bytes:
        return "⚠️ Please upload a resume first."
    if not job_title:
        return "⚠️ Please enter a desired job title."
    if not location:
        return "⚠️ Please enter a location preference."

    # --------------------------------------------------------
    # READ RESUME
    # --------------------------------------------------------
    try:
        reader = PdfReader(resume_bytes)
        resume_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                resume_text += text + "\n"

        if not resume_text.strip():
            return "⚠️ No readable text could be extracted from the resume."

    except Exception as e:
        return f"❌ RESUME READING ERROR\n{str(e)}"

    # --------------------------------------------------------
    # MASTER PROMPT
    # --------------------------------------------------------
    master_prompt = f"""
You are an elite Autonomous GenAI Job Application Agent.
Your mission is to:
1. Analyze the candidate's resume.
2. Search for REAL job postings.
3. Identify SPECIFIC job postings.
4. Select ONE strongest matching job.
5. Identify a verified application email if available.
6. Write a tailored application email.
7. Send the application email only when a legitimate
   recipient email is available.
============================================================
TARGET JOB
Desired Job Title:
{job_title}
Location Preference:
{location}
============================================================
CANDIDATE RESUME
{resume_text}
============================================================
AVAILABLE JOB SEARCH COMMAND
To search for real jobs, output:
SEARCH_JOBS: <search query>
Start with:
SEARCH_JOBS: {job_title} {location}
============================================================
WORKFLOW
STEP 1 — RESUME ANALYSIS
Analyze:
- Candidate name
- Skills
- Technical skills
- Professional skills
- Work experience
- Education
- Qualifications
- Relevant technologies
Only use information found in the resume.
NEVER invent candidate information.
STEP 2 — JOB SEARCH
Request:
SEARCH_JOBS: {job_title} {location}
STEP 3 — IDENTIFY REAL JOB POSTINGS
When the real search results are returned,
inspect them carefully.
Identify specific job postings.
For each relevant posting identify:
- Exact Job Title
- Company
- Location
- URL
STEP 4 — SELECT ONE SPECIFIC POSTING
Select ONE strongest matching job.
IMPORTANT:
The selected job MUST come directly from the
real search results.
DO NOT invent:
- Job Title
- Company
- Location
- URL
- Job description
STEP 5 — FIND APPLICATION EMAIL
Look for a legitimate application/contact email
in the available job information.
If a verified email is NOT available:
DO NOT invent one.
State:
"No verified application email was found."
STEP 6 — MATCH CANDIDATE
Compare the selected job with:
- Skills
- Experience
- Education
- Qualifications
Explain why this specific posting is the strongest match.
STEP 7 — WRITE APPLICATION
Write a professional tailored application email.
Use ONLY information contained in the resume.
Do not invent:
- Experience
- Skills
- Achievements
- Qualifications
============================================================
IMPORTANT EMAIL RULE
NEVER create a fake email such as:
hr@company.com
careers@company.com
recruitment@company.com
unless that exact email address appears in the
real information returned by the job search.
If no verified email is found:
DO NOT SEND THE EMAIL.
============================================================
FINAL REPORT FORMAT
🤖 Autonomous Job Application Report
👤 Candidate Profile
Name:
Key Skills:
Experience:
🔎 Job Search
Job Title Searched:
{job_title}
Location:
{location}
🎯 SELECTED JOB POSTING
Job Title:
Company:
Location:
Job URL:
📧 APPLICATION CONTACT
Recipient Email:
If none was found:
"No verified application email was found."
🧠 WHY THIS JOB WAS SELECTED
Explain why this SPECIFIC job posting matches
the candidate.
✉️ TAILORED APPLICATION
Subject:
Email:
📋 APPLICATION STATUS
Show one of:
- Application Sent
- Application Prepared - Not Sent
- No Suitable Job Found
============================================================
STRICT RULES
1. Never invent a job.
2. Never invent a company.
3. Never invent a URL.
4. Never invent an email address.
5. Never invent candidate information.
6. Never invent qualifications.
7. Never claim an email was sent unless the
   Python email function confirms successful delivery.
8. The selected job MUST come from the real search results.
9. The Job URL MUST come directly from the search results.
10. If there is no verified recipient email,
    prepare the application but DO NOT send it.
Begin by requesting the job search.
"""

    conversation = master_prompt

    try:
        log("🤖 Autonomous Job Application Agent started")
        log(f"🎯 Desired Job: {job_title}")
        log(f"📍 Location: {location}")

        for step in range(5):
            log(f"🔄 Agent Step {step + 1}/5")

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=conversation,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            result = response.text or ""

            # ---- DETECT JOB SEARCH REQUEST ----
            if "SEARCH_JOBS:" in result:
                search_query = result.split("SEARCH_JOBS:", 1)[1].strip()
                if "\n" in search_query:
                    search_query = search_query.split("\n")[0].strip()

                search_results = search_jobs(search_query, client, log)

                conversation = f"""
{conversation}
============================================================
GEMINI SEARCH REQUEST
{result}
============================================================
REAL SEARCH RESULTS
{search_results}
============================================================
NEXT STEP
You now have the REAL search results.
Carefully inspect them.
Select ONE specific job posting.
The selected posting MUST come from the results above.
Return:
Job Title:
Company:
Location:
Job URL:
Also identify a legitimate application/contact
email ONLY if it appears in the supplied information.
DO NOT invent:
- Job
- Company
- URL
- Email
If no verified email exists, state:
"No verified application email was found."
Then generate the complete final report.
"""
                continue

            # ---- DETECT FINAL REPORT ----
            if result.strip():
                log("✅ Final report generated")

                email_matches = re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", result)
                recipient_email = email_matches[0] if email_matches else None

                selected_job_title = job_title
                job_title_match = re.search(r"\*\*Job Title:\*\*\s*(.+)", result, re.IGNORECASE)
                if job_title_match:
                    selected_job_title = job_title_match.group(1).strip()

                company = "Unknown Company"
                company_match = re.search(r"\*\*Company:\*\*\s*(.+)", result, re.IGNORECASE)
                if company_match:
                    company = company_match.group(1).strip()

                candidate_name = "Candidate"
                name_match = re.search(r"\*\*Name:\*\*\s*(.+)", result, re.IGNORECASE)
                if name_match:
                    candidate_name = name_match.group(1).strip()

                if recipient_email:
                    log(f"📧 Verified email detected: {recipient_email}")
                    email_result = send_application_email(
                        recipient_email=recipient_email,
                        job_title=selected_job_title,
                        company=company,
                        candidate_name=candidate_name,
                        email_body=result,
                        log=log,
                    )
                    return result + "\n\n" + email_result

                log("⚠️ No verified application email found.")
                return (
                    result
                    + "\n\n📧 EMAIL STATUS\n"
                    + "⚠️ No verified recipient email was found. "
                    + "The application was prepared but NOT sent."
                )

        return (
            "⚠️ AGENT COMPLETED\n"
            "The agent reached the maximum number of processing steps "
            "without producing a final report. Please try again."
        )

    except Exception as e:
        log("❌ AGENT EXECUTION ERROR")
        traceback.print_exc()
        return f"❌ AGENT EXECUTION ERROR\nError Type:\n{type(e).__name__}\nError Message:\n{str(e)}"


# ============================================================
# STREAMLIT UI
# ============================================================

st.markdown(
    """
    <style>
    .hero-title {
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 4px;
        text-align: center;
    }
    .hero-subtitle {
        font-size: 16px;
        opacity: 0.65;
        text-align: center;
        margin-bottom: 10px;
    }
    .workflow-card {
        text-align: center;
        padding: 16px;
        border-radius: 14px;
        border: 1px solid rgba(120,120,120,0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hero-title">✨ Gemini Career Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Autonomous AI-powered job discovery and application assistant</div>',
    unsafe_allow_html=True,
)

if GEMINI_API_KEY:
    st.success("🟢 AI Agent Ready", icon="✅")
else:
    st.error(
        "🔴 GEMINI_API_KEY is not set. Add it to Streamlit secrets (Settings → Secrets) "
        "or as an environment variable before running the agent.",
        icon="⚠️",
    )

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("👤 Candidate Profile")
    st.caption("Provide your resume and job preferences.")

    resume_file = st.file_uploader("📄 Resume / CV (PDF)", type=["pdf"])
    job_title = st.text_input("🎯 Desired Job Title", placeholder="e.g. Computer Vision Engineer")
    location = st.text_input("📍 Location Preference", placeholder="e.g. Remote / London / Karachi")

    launch = st.button("🚀 Launch AI Agent", type="primary", use_container_width=True)

    st.markdown(
        """
        **Agent workflow**

        📄 Resume Analysis
        🔎 Job Discovery
        🧠 Candidate Matching
        ✍️ Application Generation
        📤 Application Workflow
        """
    )

with col_right:
    st.subheader("🧠 Agent Workspace")
    st.caption("Monitor the autonomous recruitment process.")

    status_placeholder = st.empty()
    log_placeholder = st.empty()

    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "report" not in st.session_state:
        st.session_state.report = ""

    def log(message: str):
        st.session_state.logs.append(message)
        log_placeholder.text_area(
            "Agent Execution & Audit Logs",
            value="\n".join(st.session_state.logs),
            height=420,
            disabled=True,
        )

    if not launch:
        status_placeholder.info("🟢 Agent Ready — Waiting for instructions")
        log_placeholder.text_area(
            "Agent Execution & Audit Logs",
            value="\n".join(st.session_state.logs) if st.session_state.logs else "",
            height=420,
            disabled=True,
            placeholder=(
                "Your agent activity will appear here...\n\n"
                "Example:\n"
                "→ Resume received\n"
                "→ Analyzing candidate profile\n"
                "→ Searching for relevant positions\n"
                "→ Evaluating job matches\n"
            ),
        )

if launch:
    st.session_state.logs = []
    if not GEMINI_API_KEY:
        with col_right:
            status_placeholder.error("❌ Missing GEMINI_API_KEY — cannot start agent.")
    else:
        with col_right:
            status_placeholder.warning("🟡 Agent running...")
        client = get_client(GEMINI_API_KEY)
        with st.spinner("Agent is analyzing your resume and searching for jobs..."):
            report = run_autonomous_agent(resume_file, job_title, location, client, log)
        st.session_state.report = report
        with col_right:
            status_placeholder.success("✅ Agent finished")
            st.markdown("### 📋 Final Report")
            st.markdown(report)

# ============================================================
# WORKFLOW SECTION
# ============================================================

st.markdown("### ⚡ AI Agent Workflow")
w1, w2, w3, w4 = st.columns(4)

with w1:
    st.markdown(
        '<div class="workflow-card">📄<br><b>Resume Intelligence</b>'
        '<br><span style="font-size:12px;opacity:0.6">Extract skills, experience and qualifications</span></div>',
        unsafe_allow_html=True,
    )
with w2:
    st.markdown(
        '<div class="workflow-card">🔎<br><b>Job Discovery</b>'
        '<br><span style="font-size:12px;opacity:0.6">Identify opportunities matching your profile</span></div>',
        unsafe_allow_html=True,
    )
with w3:
    st.markdown(
        '<div class="workflow-card">🧠<br><b>AI Matching</b>'
        '<br><span style="font-size:12px;opacity:0.6">Evaluate candidate-to-job relevance</span></div>',
        unsafe_allow_html=True,
    )
with w4:
    st.markdown(
        '<div class="workflow-card">✍️<br><b>Application Agent</b>'
        '<br><span style="font-size:12px;opacity:0.6">Generate and prepare applications</span></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div style="text-align:center; padding:25px 0 5px 0; font-size:12px; opacity:0.5;">'
    "Gemini Career Agent • Generative AI</div>",
    unsafe_allow_html=True,
)
