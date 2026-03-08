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
    prompt = f"""
You are an expert resume writer specializing in high school students with little to no work experience.
Your job is to take raw input from a student and turn it into a polished, professional resume.

Transform their activities, clubs, and experiences into strong action-oriented bullet points.
Make them sound impressive without exaggerating or fabricating anything.
Keep bullet points concise (1-2 lines each).
Use strong action verbs (Led, Organized, Developed, Achieved, etc.)

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

Return a JSON object with exactly these keys:
{{
  "objective": "A 2-sentence professional objective statement tailored to what they are applying for",
  "education": ["bullet point 1", "bullet point 2"],
  "activities": ["bullet point 1", "bullet point 2", "..."],
  "experience": ["bullet point 1", "..."],
  "skills": ["skill 1", "skill 2", "..."]
}}

If there is no experience, return an empty list for experience.
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert resume writer for high school students."},
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


def _pdf_classic(data: dict, resume: dict) -> io.BytesIO:
    """Centered header, purple accent lines."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=0.75*inch, leftMargin=0.75*inch,
        topMargin=0.65*inch, bottomMargin=0.65*inch
    )
    ACCENT = colors.HexColor("#4f46e5")
    DARK   = colors.HexColor("#111111")
    GRAY   = colors.HexColor("#555555")

    name_style    = ParagraphStyle("CName", fontSize=22, fontName="Helvetica-Bold", textColor=DARK, alignment=TA_CENTER, spaceAfter=3, leading=26)
    sub_style     = ParagraphStyle("CSub",  fontSize=10, fontName="Helvetica",      textColor=GRAY, alignment=TA_CENTER, spaceAfter=10, leading=14)
    section_style = ParagraphStyle("CSec",  fontSize=9,  fontName="Helvetica-Bold", textColor=ACCENT, spaceBefore=14, spaceAfter=2, leading=12, letterSpacing=1.2)
    bullet_style  = ParagraphStyle("CBul",  fontSize=10, fontName="Helvetica",      textColor=DARK, spaceAfter=3, leading=14, leftIndent=14, firstLineIndent=-10)
    obj_style     = ParagraphStyle("CObj",  fontSize=10, fontName="Helvetica-Oblique", textColor=GRAY, spaceAfter=6, leading=15)
    skills_style  = ParagraphStyle("CSkl",  fontSize=10, fontName="Helvetica",      textColor=DARK, spaceAfter=3, leading=14, leftIndent=4)

    def sec(title):
        return [Paragraph(title, section_style), HRFlowable(width="100%", thickness=0.75, color=ACCENT, spaceAfter=4)]

    def bul(items):
        return [Paragraph(f"\u2022\u2002{b}", bullet_style) for b in items]

    content = []
    sub_parts = [data.get("school", ""), f"Grade {data.get('grade', '')}"]
    if data.get("gpa"):
        sub_parts.append(f"GPA: {data.get('gpa')}")

    content.append(Paragraph(data.get("name", ""), name_style))
    content.append(Paragraph("  \u2022  ".join(sub_parts), sub_style))
    content.append(HRFlowable(width="100%", thickness=2, color=ACCENT, spaceAfter=4))

    content += sec("OBJECTIVE")
    content.append(Paragraph(resume.get("objective", ""), obj_style))
    content += sec("EDUCATION")
    content += bul(resume.get("education", []))
    if resume.get("activities"):
        content += sec("ACTIVITIES & LEADERSHIP")
        content += bul(resume.get("activities", []))
    if resume.get("experience"):
        content += sec("EXPERIENCE")
        content += bul(resume.get("experience", []))
    if resume.get("skills"):
        content += sec("SKILLS")
        content.append(Paragraph(", ".join(resume.get("skills", [])), skills_style))

    doc.build(content)
    buffer.seek(0)
    return buffer


def _pdf_modern(data: dict, resume: dict) -> io.BytesIO:
    """Dark sidebar (name + skills) on the left, content on the right."""
    buffer = io.BytesIO()
    PAGE_W, PAGE_H = letter
    SIDEBAR_W = 2.15 * inch

    SIDEBAR_BG = colors.HexColor("#1e293b")
    ACCENT     = colors.HexColor("#818cf8")
    WHITE      = colors.white
    LIGHT      = colors.HexColor("#94a3b8")
    BLACK      = colors.HexColor("#111111")
    MID_GRAY   = colors.HexColor("#555555")
    LINE_GRAY  = colors.HexColor("#d1d5db")

    def draw_sidebar(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(SIDEBAR_BG)
        canvas.rect(0, 0, SIDEBAR_W, PAGE_H, fill=1, stroke=0)
        canvas.restoreState()

    sidebar_frame = Frame(
        0.15*inch, 0.4*inch,
        SIDEBAR_W - 0.15*inch, PAGE_H - 0.8*inch,
        leftPadding=10, rightPadding=8, topPadding=22, bottomPadding=10,
        id="sidebar"
    )
    main_frame = Frame(
        SIDEBAR_W + 0.3*inch, 0.4*inch,
        PAGE_W - SIDEBAR_W - 0.75*inch, PAGE_H - 0.8*inch,
        leftPadding=6, rightPadding=10, topPadding=22, bottomPadding=10,
        id="main"
    )
    doc = BaseDocTemplate(buffer, pagesize=letter, leftMargin=0, rightMargin=0, topMargin=0, bottomMargin=0)
    doc.addPageTemplates([PageTemplate(id="TwoCol", frames=[sidebar_frame, main_frame], onPage=draw_sidebar)])

    sb_name = ParagraphStyle("SBN", fontSize=17, fontName="Helvetica-Bold", textColor=WHITE,      leading=20, spaceAfter=6)
    sb_sub  = ParagraphStyle("SBS", fontSize=8.5, fontName="Helvetica",     textColor=LIGHT,     leading=12, spaceAfter=3)
    sb_sec  = ParagraphStyle("SBC", fontSize=8,   fontName="Helvetica-Bold",textColor=ACCENT,    leading=10, spaceBefore=14, spaceAfter=4, letterSpacing=1.2)
    sb_bul  = ParagraphStyle("SBB", fontSize=9,   fontName="Helvetica",     textColor=WHITE,     leading=13, spaceAfter=2)
    mn_sec  = ParagraphStyle("MNS", fontSize=9,   fontName="Helvetica-Bold",textColor=colors.HexColor("#4f46e5"), leading=11, spaceBefore=12, spaceAfter=2, letterSpacing=1.0)
    mn_bul  = ParagraphStyle("MNB", fontSize=10,  fontName="Helvetica",     textColor=BLACK,     leading=14, spaceAfter=3, leftIndent=12, firstLineIndent=-8)
    mn_obj  = ParagraphStyle("MNO", fontSize=10,  fontName="Helvetica-Oblique", textColor=MID_GRAY, leading=15, spaceAfter=6)

    def mn_sec_hdr(title):
        return [Paragraph(title, mn_sec), HRFlowable(width="100%", thickness=0.5, color=LINE_GRAY, spaceAfter=4)]

    def mn_buls(items):
        return [Paragraph(f"\u2022\u2002{b}", mn_bul) for b in items]

    content = []

    # Sidebar
    content.append(Paragraph(data.get("name", ""), sb_name))
    content.append(Paragraph(data.get("school", ""), sb_sub))
    content.append(Paragraph(f"Grade {data.get('grade', '')}", sb_sub))
    if data.get("gpa"):
        content.append(Paragraph(f"GPA: {data.get('gpa')}", sb_sub))
    if resume.get("skills"):
        content.append(Spacer(1, 6))
        content.append(HRFlowable(width="100%", thickness=0.5, color=ACCENT))
        content.append(Paragraph("SKILLS", sb_sec))
        for skill in resume.get("skills", []):
            content.append(Paragraph(f"\u2022 {skill}", sb_bul))
    content.append(FrameBreak())

    # Main
    content += mn_sec_hdr("OBJECTIVE")
    content.append(Paragraph(resume.get("objective", ""), mn_obj))
    content += mn_sec_hdr("EDUCATION")
    content += mn_buls(resume.get("education", []))
    if resume.get("activities"):
        content += mn_sec_hdr("ACTIVITIES & LEADERSHIP")
        content += mn_buls(resume.get("activities", []))
    if resume.get("experience"):
        content += mn_sec_hdr("EXPERIENCE")
        content += mn_buls(resume.get("experience", []))

    doc.build(content)
    buffer.seek(0)
    return buffer


def _pdf_executive(data: dict, resume: dict) -> io.BytesIO:
    """Traditional black & white, left-aligned, no color."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=0.9*inch, leftMargin=0.9*inch,
        topMargin=0.75*inch, bottomMargin=0.75*inch
    )
    BLACK     = colors.HexColor("#111111")
    DARK_GRAY = colors.HexColor("#333333")
    GRAY      = colors.HexColor("#555555")

    name_style = ParagraphStyle("EName", fontSize=24, fontName="Helvetica-Bold", textColor=BLACK,     spaceAfter=2,  leading=28)
    sub_style  = ParagraphStyle("ESub",  fontSize=10, fontName="Helvetica",      textColor=GRAY,     spaceAfter=10, leading=14)
    sec_style  = ParagraphStyle("ESec",  fontSize=10, fontName="Helvetica-Bold", textColor=BLACK,    spaceBefore=14, spaceAfter=2, leading=13, letterSpacing=0.5)
    bul_style  = ParagraphStyle("EBul",  fontSize=10, fontName="Helvetica",      textColor=BLACK,    spaceAfter=3,  leading=14, leftIndent=14, firstLineIndent=-10)
    obj_style  = ParagraphStyle("EObj",  fontSize=10, fontName="Helvetica",      textColor=DARK_GRAY, spaceAfter=6, leading=15)
    skl_style  = ParagraphStyle("ESkl",  fontSize=10, fontName="Helvetica",      textColor=BLACK,    spaceAfter=3,  leading=14, leftIndent=4)

    def sec(title):
        return [Paragraph(title, sec_style), HRFlowable(width="100%", thickness=1.5, color=BLACK, spaceAfter=5)]

    def bul(items):
        return [Paragraph(f"\u2022\u2002{b}", bul_style) for b in items]

    content = []
    sub_parts = [data.get("school", ""), f"Grade {data.get('grade', '')}"]
    if data.get("gpa"):
        sub_parts.append(f"GPA: {data.get('gpa')}")

    content.append(Paragraph(data.get("name", ""), name_style))
    content.append(Paragraph("  |  ".join(sub_parts), sub_style))
    content.append(HRFlowable(width="100%", thickness=2, color=BLACK, spaceAfter=4))

    content += sec("OBJECTIVE")
    content.append(Paragraph(resume.get("objective", ""), obj_style))
    content += sec("EDUCATION")
    content += bul(resume.get("education", []))
    if resume.get("activities"):
        content += sec("ACTIVITIES & LEADERSHIP")
        content += bul(resume.get("activities", []))
    if resume.get("experience"):
        content += sec("EXPERIENCE")
        content += bul(resume.get("experience", []))
    if resume.get("skills"):
        content += sec("SKILLS")
        content.append(Paragraph(", ".join(resume.get("skills", [])), skl_style))

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
