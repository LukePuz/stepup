import os
import json
import io
from flask import Flask, render_template, request, send_file
from openai import OpenAI
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, BaseDocTemplate, Frame, PageTemplate, FrameBreak,
    Paragraph, Spacer, HRFlowable, Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import requests as http_requests

load_dotenv(override=True)

app = Flask(__name__)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://"
)

# Global daily cap across all users
daily_count = {"date": None, "count": 0}
DAILY_GLOBAL_LIMIT = 50

def check_global_limit():
    from datetime import date
    today = str(date.today())
    if daily_count["date"] != today:
        daily_count["date"] = today
        daily_count["count"] = 0
    if daily_count["count"] >= DAILY_GLOBAL_LIMIT:
        return False
    daily_count["count"] += 1
    return True


def generate_resume(data: dict) -> dict:
    prompt = f"""You are an expert resume writer for high school students with little or no work experience.
Transform the student's raw input into a polished, professional resume.

Student Input:
- Name: {data.get('name')}
- School: {data.get('school')}
- Grade: {data.get('grade')}
- GPA: {data.get('gpa')}
- Applying for: {data.get('applying_for')}
- Activities & Clubs: {data.get('activities')}
- Skills: {data.get('skills')}
- Experience (jobs/volunteering): {data.get('experience') or 'None'}
- Extra info: {data.get('extra') or 'None'}

Return a JSON object with EXACTLY this structure — do not add extra keys:
{{
  "objective": "A 2-sentence professional summary tailored to what they are applying for",
  "education": [
    {{
      "title": "School name",
      "meta": "Grade level | GPA if provided",
      "bullets": ["Relevant academic detail or achievement (1-2 max)"]
    }}
  ],
  "activities": [
    {{
      "title": "Club, sport, or activity name",
      "meta": "Role | Year or date range",
      "bullets": ["Action-verb impact bullet", "Second bullet if relevant"]
    }}
  ],
  "experience": [
    {{
      "title": "Employer or organization name",
      "meta": "Role | Date range",
      "bullets": ["Action-verb bullet describing contribution"]
    }}
  ],
  "skills": {{
    "Category Name": ["skill 1", "skill 2"],
    "Another Category": ["skill 3", "skill 4"]
  }}
}}

Rules:
- Use strong action verbs: Led, Built, Developed, Managed, Organized, Achieved, Designed, Coordinated
- Keep each bullet to one concise line
- Group skills into 2-4 meaningful categories based on the student's background (e.g. Technical, Tools, Languages, Soft Skills)
- If no experience provided, return "experience": []
- Each activity/experience entry: 1-3 bullets max
- Do not fabricate or exaggerate information
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert resume writer for high school students. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)


def build_pdf(data: dict, resume: dict, template: str = "classic") -> io.BytesIO:
    if template == "modern":
        return _pdf_modern(data, resume)
    elif template == "executive":
        return _pdf_executive(data, resume)
    else:
        return _pdf_classic(data, resume)


# ── Shared PDF utilities ─────────────────────────────────────────────────────

def _normalize_skills(skills):
    """Accept list or dict from AI; always return a dict."""
    if isinstance(skills, dict):
        return skills
    if isinstance(skills, list):
        return {"Skills": skills}
    return {}


def _pdf_entries(entries, s_title, s_meta, s_bullet):
    """Render a list of structured {title, meta, bullets} entry dicts."""
    out = []
    entries = entries if isinstance(entries, list) else []
    for i, e in enumerate(entries):
        if isinstance(e, str):
            out.append(Paragraph(f"\u2022\u2002{e}", s_bullet))
            continue
        out.append(Paragraph(e.get("title", ""), s_title))
        if e.get("meta"):
            out.append(Paragraph(e["meta"], s_meta))
        for b in e.get("bullets", []):
            out.append(Paragraph(f"\u2022\u2002{b}", s_bullet))
        if i < len(entries) - 1:
            out.append(Spacer(1, 10))
    return out


def _pdf_skills_grid(skills_raw, s_cat, s_item, col_w):
    """Render categorized skills in a two-column table."""
    skills = _normalize_skills(skills_raw)
    cats = list(skills.items())
    if not cats:
        return []
    half = (len(cats) + 1) // 2
    left_cats, right_cats = cats[:half], cats[half:]

    def make_col(pairs):
        cell = []
        for cat_name, items in pairs:
            cell.append(Paragraph(cat_name, s_cat))
            for item in (items or []):
                cell.append(Paragraph(item, s_item))
            cell.append(Spacer(1, 5))
        return cell

    t = Table([[make_col(left_cats), make_col(right_cats)]], colWidths=[col_w, col_w])
    t.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return [t, Spacer(1, 4)]


# ── PDF Templates ─────────────────────────────────────────────────────────────

def _pdf_classic(data: dict, resume: dict) -> io.BytesIO:
    """Professional: navy banner header, blue section titles, entry hierarchy."""
    buffer = io.BytesIO()
    PAGE_W, PAGE_H = letter
    HEADER_H = 1.1 * inch
    NAVY   = colors.HexColor("#1B3A6B")
    ACCENT = colors.HexColor("#2563EB")
    DARK   = colors.HexColor("#1F2937")
    GRAY   = colors.HexColor("#6B7280")
    WHITE  = colors.white
    LM = RM = 0.75 * inch
    usable_w = PAGE_W - LM - RM

    contact_parts = [data.get("school", ""), f"Grade {data.get('grade', '')}"]
    if data.get("gpa"):
        contact_parts.append(f"GPA: {data['gpa']}")
    name_text    = data.get("name", "")
    contact_text = "  \u2022  ".join(contact_parts)

    def draw_header(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, PAGE_H - HEADER_H, PAGE_W, HEADER_H, fill=1, stroke=0)
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 26)
        canvas.drawCentredString(PAGE_W / 2, PAGE_H - 0.56 * inch, name_text)
        canvas.setFont("Helvetica", 9)
        canvas.drawCentredString(PAGE_W / 2, PAGE_H - 0.82 * inch, contact_text)
        canvas.restoreState()

    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=RM, leftMargin=LM,
        topMargin=HEADER_H + 0.25 * inch, bottomMargin=0.75 * inch
    )

    S_SEC   = ParagraphStyle("cSec",   fontSize=10, fontName="Helvetica-Bold", textColor=ACCENT,
                              spaceBefore=18, spaceAfter=6, leading=12)
    S_ETIT  = ParagraphStyle("cETit",  fontSize=11, fontName="Helvetica-Bold", textColor=DARK,
                              spaceAfter=1, leading=14)
    S_EMETA = ParagraphStyle("cEMeta", fontSize=9,  fontName="Helvetica", textColor=GRAY,
                              spaceAfter=3, leading=12)
    S_BUL   = ParagraphStyle("cBul",   fontSize=9.5, fontName="Helvetica", textColor=DARK,
                              spaceAfter=4, leading=13, leftIndent=12, firstLineIndent=-8)
    S_OBJ   = ParagraphStyle("cObj",   fontSize=9.5, fontName="Helvetica", textColor=GRAY,
                              spaceAfter=6, leading=14)
    S_SCAT  = ParagraphStyle("cScat",  fontSize=9.5, fontName="Helvetica-Bold", textColor=DARK,
                              spaceAfter=2, spaceBefore=4, leading=13)
    S_SITEM = ParagraphStyle("cSitem", fontSize=9.5, fontName="Helvetica", textColor=DARK,
                              spaceAfter=2, leading=12, leftIndent=8)

    def sec(title):
        return [Paragraph(title.upper(), S_SEC)]

    content = []
    if resume.get("objective"):
        content += sec("Summary")
        content.append(Paragraph(resume["objective"], S_OBJ))
    if resume.get("skills"):
        content += sec("Skills")
        content += _pdf_skills_grid(resume["skills"], S_SCAT, S_SITEM, usable_w / 2)
    if resume.get("experience"):
        content += sec("Experience")
        content += _pdf_entries(resume["experience"], S_ETIT, S_EMETA, S_BUL)
    content += sec("Education")
    content += _pdf_entries(resume.get("education", []), S_ETIT, S_EMETA, S_BUL)
    if resume.get("activities"):
        content += sec("Activities & Leadership")
        content += _pdf_entries(resume["activities"], S_ETIT, S_EMETA, S_BUL)

    doc.build(content, onFirstPage=draw_header, onLaterPages=draw_header)
    buffer.seek(0)
    return buffer


def _pdf_modern(data: dict, resume: dict) -> io.BytesIO:
    """Minimal: large left-aligned black name, no color accents, entry hierarchy."""
    buffer = io.BytesIO()
    PAGE_W, _ = letter
    DARK = colors.HexColor("#111111")
    GRAY = colors.HexColor("#555555")
    LM = RM = 0.75 * inch
    usable_w = PAGE_W - LM - RM

    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=RM, leftMargin=LM,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch
    )

    S_NAME  = ParagraphStyle("mName",  fontSize=28, fontName="Helvetica-Bold", textColor=DARK,
                              spaceAfter=4, leading=32)
    S_CON   = ParagraphStyle("mCon",   fontSize=9.5, fontName="Helvetica", textColor=GRAY,
                              spaceAfter=16, leading=13)
    S_SEC   = ParagraphStyle("mSec",   fontSize=10, fontName="Helvetica-Bold", textColor=DARK,
                              spaceBefore=18, spaceAfter=6, leading=12)
    S_ETIT  = ParagraphStyle("mETit",  fontSize=11, fontName="Helvetica-Bold", textColor=DARK,
                              spaceAfter=1, leading=14)
    S_EMETA = ParagraphStyle("mEMeta", fontSize=9,  fontName="Helvetica", textColor=GRAY,
                              spaceAfter=3, leading=12)
    S_BUL   = ParagraphStyle("mBul",   fontSize=9.5, fontName="Helvetica", textColor=DARK,
                              spaceAfter=4, leading=13, leftIndent=12, firstLineIndent=-8)
    S_OBJ   = ParagraphStyle("mObj",   fontSize=9.5, fontName="Helvetica", textColor=GRAY,
                              spaceAfter=6, leading=14)
    S_SCAT  = ParagraphStyle("mScat",  fontSize=9.5, fontName="Helvetica-Bold", textColor=DARK,
                              spaceAfter=2, spaceBefore=4, leading=13)
    S_SITEM = ParagraphStyle("mSitem", fontSize=9.5, fontName="Helvetica", textColor=DARK,
                              spaceAfter=2, leading=12, leftIndent=8)

    def sec(title):
        return [Paragraph(title.upper(), S_SEC)]

    contact_parts = [data.get("school", ""), f"Grade {data.get('grade', '')}"]
    if data.get("gpa"):
        contact_parts.append(f"GPA: {data['gpa']}")

    content = []
    content.append(Paragraph(data.get("name", ""), S_NAME))
    content.append(Paragraph("  \u2022  ".join(contact_parts), S_CON))
    if resume.get("objective"):
        content += sec("Summary")
        content.append(Paragraph(resume["objective"], S_OBJ))
    if resume.get("skills"):
        content += sec("Skills")
        content += _pdf_skills_grid(resume["skills"], S_SCAT, S_SITEM, usable_w / 2)
    if resume.get("experience"):
        content += sec("Experience")
        content += _pdf_entries(resume["experience"], S_ETIT, S_EMETA, S_BUL)
    content += sec("Education")
    content += _pdf_entries(resume.get("education", []), S_ETIT, S_EMETA, S_BUL)
    if resume.get("activities"):
        content += sec("Activities & Leadership")
        content += _pdf_entries(resume["activities"], S_ETIT, S_EMETA, S_BUL)

    doc.build(content)
    buffer.seek(0)
    return buffer


def _pdf_executive(data: dict, resume: dict) -> io.BytesIO:
    """Elegant: centered uppercase name, decorative section headers, entry hierarchy."""
    from reportlab.platypus import Flowable as _Flowable

    buffer = io.BytesIO()
    PAGE_W, _ = letter
    DARK = colors.HexColor("#1A1A1A")
    GRAY = colors.HexColor("#666666")
    LINE = colors.HexColor("#4A4A4A")
    LM = RM = 0.75 * inch
    usable_w = PAGE_W - LM - RM

    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=RM, leftMargin=LM,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch
    )

    S_NAME  = ParagraphStyle("eName",  fontSize=24, fontName="Helvetica-Bold", textColor=DARK,
                              alignment=TA_CENTER, spaceAfter=4, leading=28, letterSpacing=3.0)
    S_CON   = ParagraphStyle("eCon",   fontSize=9,  fontName="Helvetica", textColor=GRAY,
                              alignment=TA_CENTER, spaceAfter=14, leading=13)
    S_ETIT  = ParagraphStyle("eETit",  fontSize=11, fontName="Helvetica-Bold", textColor=DARK,
                              spaceAfter=1, leading=14)
    S_EMETA = ParagraphStyle("eEMeta", fontSize=9,  fontName="Helvetica", textColor=GRAY,
                              spaceAfter=3, leading=12)
    S_BUL   = ParagraphStyle("eBul",   fontSize=9.5, fontName="Helvetica", textColor=DARK,
                              spaceAfter=4, leading=13, leftIndent=12, firstLineIndent=-8)
    S_OBJ   = ParagraphStyle("eObj",   fontSize=9.5, fontName="Helvetica", textColor=GRAY,
                              alignment=TA_CENTER, spaceAfter=6, leading=15)
    S_SCAT  = ParagraphStyle("eScat",  fontSize=9.5, fontName="Helvetica-Bold", textColor=DARK,
                              spaceAfter=2, spaceBefore=4, leading=13)
    S_SITEM = ParagraphStyle("eSitem", fontSize=9.5, fontName="Helvetica", textColor=DARK,
                              spaceAfter=2, leading=12, leftIndent=8)

    class DecorativeHeader(_Flowable):
        def __init__(self, title, width, color):
            _Flowable.__init__(self)
            self.title = title.upper()
            self.avail_w = width
            self.color = color
            self.height = 22

        def draw(self):
            c = self.canv
            c.saveState()
            fs = 9
            text_w = c.stringWidth(self.title, "Helvetica-Bold", fs)
            cx = self.avail_w / 2
            tx = cx - text_w / 2
            ty = 5
            ly = ty + fs * 0.5
            c.setFillColor(self.color)
            c.setFont("Helvetica-Bold", fs)
            c.drawString(tx, ty, self.title)
            c.setStrokeColor(self.color)
            c.setLineWidth(0.5)
            pad = 8
            if tx > pad:
                c.line(0, ly, tx - pad, ly)
                c.line(tx + text_w + pad, ly, self.avail_w, ly)
            c.restoreState()

        def wrap(self, availWidth, availHeight):
            return (self.avail_w, self.height)

    def sec(title):
        return [Spacer(1, 6), DecorativeHeader(title, usable_w, LINE), Spacer(1, 6)]

    contact_parts = [data.get("school", ""), f"Grade {data.get('grade', '')}"]
    if data.get("gpa"):
        contact_parts.append(f"GPA: {data['gpa']}")

    content = []
    content.append(Paragraph(data.get("name", "").upper(), S_NAME))
    content.append(Paragraph("  |  ".join(contact_parts), S_CON))
    if resume.get("objective"):
        content += sec("Summary")
        content.append(Paragraph(resume["objective"], S_OBJ))
    if resume.get("skills"):
        content += sec("Skills")
        content += _pdf_skills_grid(resume["skills"], S_SCAT, S_SITEM, usable_w / 2)
    if resume.get("experience"):
        content += sec("Experience")
        content += _pdf_entries(resume["experience"], S_ETIT, S_EMETA, S_BUL)
    content += sec("Education")
    content += _pdf_entries(resume.get("education", []), S_ETIT, S_EMETA, S_BUL)
    if resume.get("activities"):
        content += sec("Activities & Leadership")
        content += _pdf_entries(resume["activities"], S_ETIT, S_EMETA, S_BUL)

    doc.build(content)
    buffer.seek(0)
    return buffer


@app.route("/")
def index():
    count = 0
    try:
        r = http_requests.get(
            f"{os.getenv('SUPABASE_URL')}/rest/v1/resume-generations?select=*",
            headers={
                "apikey": os.getenv("SUPABASE_KEY"),
                "Authorization": f"Bearer {os.getenv('SUPABASE_KEY')}",
                "Prefer": "count=exact"
            }
        )
        content_range = r.headers.get("Content-Range", "0-0/0")
        count = int(content_range.split("/")[-1])
    except Exception:
        pass
    return render_template("index.html", count=count)


@app.route("/build", methods=["GET", "POST"])
@limiter.limit("3 per day", methods=["POST"])
def build():
    if request.method == "POST":
        if not check_global_limit():
            return render_template("limit.html"), 429
        data = {
            "name": request.form.get("name"),
            "school": request.form.get("school"),
            "grade": request.form.get("grade"),
            "gpa": request.form.get("gpa"),
            "applying_for": request.form.get("applying_for"),
            "activities": request.form.get("activities"),
            "skills": request.form.get("skills"),
            "experience": request.form.get("experience"),
            "extra": request.form.get("extra"),
        }
        resume = generate_resume(data)
        return render_template("result.html", data=data, resume=resume)
    return render_template("build.html", prefill=request.args)


@app.route("/download", methods=["POST"])
def download():
    data = json.loads(request.form.get("data"))
    resume = json.loads(request.form.get("resume"))
    template = request.form.get("template", "classic")
    try:
        http_requests.post(
            f"{os.getenv('SUPABASE_URL')}/rest/v1/resume-generations",
            json={"template": template},
            headers={
                "apikey": os.getenv("SUPABASE_KEY"),
                "Authorization": f"Bearer {os.getenv('SUPABASE_KEY')}",
                "Content-Type": "application/json"
            }
        )
    except Exception:
        pass
    pdf = build_pdf(data, resume, template)
    return send_file(
        pdf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"{data.get('name', 'resume').replace(' ', '_')}_resume.pdf"
    )


@app.errorhandler(429)
def rate_limit_exceeded(e):
    return render_template("limit.html"), 429


if __name__ == "__main__":
    app.run(debug=True)
