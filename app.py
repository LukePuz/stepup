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


def _fmt_phone(phone: str) -> str:
    digits = ''.join(c for c in (phone or '') if c.isdigit())
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    if len(digits) == 11 and digits[0] == '1':
        return f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    return phone or ''

app.jinja_env.filters['fmt_phone'] = _fmt_phone


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
    elif template == "lateral":
        return _pdf_lateral(data, resume)
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
    contact2_parts = []
    if data.get("email"):
        contact2_parts.append(data["email"])
    if data.get("phone"):
        contact2_parts.append(_fmt_phone(data["phone"]))
    contact2_text = "  \u2022  ".join(contact2_parts)

    def draw_header(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, PAGE_H - HEADER_H, PAGE_W, HEADER_H, fill=1, stroke=0)
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 24)
        canvas.drawCentredString(PAGE_W / 2, PAGE_H - 0.50 * inch, name_text)
        canvas.setFont("Helvetica", 9)
        canvas.drawCentredString(PAGE_W / 2, PAGE_H - 0.72 * inch, contact_text)
        if contact2_text:
            canvas.drawCentredString(PAGE_W / 2, PAGE_H - 0.90 * inch, contact2_text)
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
    """Meridian: navy banner, two-column body, right-aligned dates, full-page fill."""
    from reportlab.platypus import Flowable as _Flowable

    buffer = io.BytesIO()
    PAGE_W, PAGE_H = letter
    HEADER_H = 1.1 * inch
    NAVY    = colors.HexColor("#1B2A4A")
    DARK    = colors.HexColor("#2D2D2D")
    GRAY    = colors.HexColor("#888888")
    LGRAY   = colors.HexColor("#94A9C4")
    SBARBG  = colors.HexColor("#F4F5F7")
    DIVIDER = colors.HexColor("#D8DCE4")
    LM = RM = 0.68 * inch
    usable_w  = PAGE_W - LM - RM
    sidebar_w = usable_w * 0.32 - 8
    main_w    = usable_w * 0.68 - 8

    contact_parts = [data.get("school", ""), f"Grade {data.get('grade', '')}"]
    if data.get("gpa"):
        contact_parts.append(f"GPA: {data['gpa']}")
    contact2_parts = []
    if data.get("email"):
        contact2_parts.append(data["email"])
    if data.get("phone"):
        contact2_parts.append(_fmt_phone(data["phone"]))
    name_text     = data.get("name", "")
    contact_text  = "   |   ".join(contact_parts)
    contact2_text = "   |   ".join(contact2_parts)

    def draw_header(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, PAGE_H - HEADER_H, PAGE_W, HEADER_H, fill=1, stroke=0)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 24)
        canvas.drawCentredString(PAGE_W / 2, PAGE_H - 0.46 * inch, name_text)
        canvas.setFillColor(LGRAY)
        canvas.setFont("Helvetica", 9)
        canvas.drawCentredString(PAGE_W / 2, PAGE_H - 0.72 * inch, contact_text)
        if contact2_text:
            canvas.drawCentredString(PAGE_W / 2, PAGE_H - 0.91 * inch, contact2_text)
        canvas.restoreState()

    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=RM, leftMargin=LM,
        topMargin=HEADER_H + 0.20 * inch, bottomMargin=0.68 * inch
    )

    S_ETIT  = ParagraphStyle("mETit",  fontSize=11,   fontName="Helvetica-Bold", textColor=NAVY,
                              spaceAfter=2,  leading=14)
    S_EDATE = ParagraphStyle("mEDate", fontSize=9,    fontName="Helvetica",      textColor=GRAY,
                              alignment=2,  leading=14)   # TA_RIGHT = 2
    S_EMETA = ParagraphStyle("mEMeta", fontSize=9.5,  fontName="Helvetica",      textColor=GRAY,
                              spaceAfter=4, leading=13)
    S_BUL   = ParagraphStyle("mBul",   fontSize=10,   fontName="Helvetica",      textColor=DARK,
                              spaceAfter=5, leading=15, leftIndent=13, firstLineIndent=-9)
    S_OBJ   = ParagraphStyle("mObj",   fontSize=10,   fontName="Helvetica",      textColor=DARK,
                              spaceAfter=8, leading=16)
    S_SCAT  = ParagraphStyle("mScat",  fontSize=9.5,  fontName="Helvetica-Bold", textColor=DARK,
                              spaceAfter=3, spaceBefore=10, leading=13)
    S_SITEM = ParagraphStyle("mSitem", fontSize=9.5,  fontName="Helvetica",      textColor=DARK,
                              spaceAfter=3, leading=13)

    class AccentTitle(_Flowable):
        """Section title with a left navy accent bar."""
        def __init__(self, title, col_w, first=False):
            _Flowable.__init__(self)
            self.title = title.upper()
            self.col_w = col_w
            self._pad  = 0 if first else 18
            self._h    = 20

        def wrap(self, aw, ah):
            return (self.col_w, self._h + self._pad)

        def draw(self):
            c = self.canv
            c.saveState()
            c.translate(0, self._pad)
            c.setFillColor(NAVY)
            c.rect(0, 3, 3, 13, fill=1, stroke=0)
            c.setFont("Helvetica-Bold", 9)
            c.drawString(10, 5, self.title)
            c.restoreState()

    DATE_COL = 0.9 * inch

    def _meridian_main_entries(entries):
        """Entries for main column: entry title + right-aligned date on one row."""
        out = []
        entries = entries if isinstance(entries, list) else []
        for i, e in enumerate(entries):
            if isinstance(e, str):
                out.append(Paragraph(f"\u2022\u2002{e}", S_BUL))
                continue
            title = e.get("title", "")
            meta  = e.get("meta", "")
            date_str = ""
            role_str = meta
            if " | " in meta:
                parts    = meta.split(" | ")
                date_str = parts[-1].strip()
                role_str = " | ".join(parts[:-1]).strip()
            title_w = main_w - DATE_COL
            if date_str:
                row = Table(
                    [[Paragraph(title, S_ETIT), Paragraph(date_str, S_EDATE)]],
                    colWidths=[title_w, DATE_COL]
                )
                row.setStyle(TableStyle([
                    ("VALIGN",        (0, 0), (-1, -1), "BOTTOM"),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
                    ("TOPPADDING",    (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]))
                out.append(row)
            else:
                out.append(Paragraph(title, S_ETIT))
            if role_str:
                out.append(Paragraph(role_str, S_EMETA))
            for b in e.get("bullets", []):
                out.append(Paragraph(f"\u2022\u2002{b}", S_BUL))
            if i < len(entries) - 1:
                out.append(Spacer(1, 12))
        return out

    def sidebar_skills(skills_raw, first):
        skills = _normalize_skills(skills_raw)
        out = []
        for i, (cat_name, items) in enumerate(skills.items()):
            if i == 0:
                out.append(AccentTitle("Skills", sidebar_w, first=first))
            out.append(Paragraph(cat_name, S_SCAT))
            for item in (items or []):
                out.append(Paragraph(item, S_SITEM))
        return out

    # Sidebar: Skills + Education
    has_skills = bool(resume.get("skills"))
    sidebar = []
    if has_skills:
        sidebar += sidebar_skills(resume["skills"], first=True)
    sidebar.append(AccentTitle("Education", sidebar_w, first=not has_skills))
    sidebar += _pdf_entries(resume.get("education", []), S_ETIT, S_EMETA, S_BUL)

    # Main: Summary + Experience + Activities
    has_obj = bool(resume.get("objective"))
    has_exp = bool(resume.get("experience"))
    main = []
    if has_obj:
        main.append(AccentTitle("Summary", main_w, first=True))
        main.append(Paragraph(resume["objective"], S_OBJ))
    if has_exp:
        main.append(AccentTitle("Experience", main_w, first=not has_obj))
        main += _meridian_main_entries(resume["experience"])
    if resume.get("activities"):
        first_main = not has_obj and not has_exp
        main.append(AccentTitle("Activities & Leadership", main_w, first=first_main))
        main += _meridian_main_entries(resume["activities"])

    body = Table(
        [[sidebar, main]],
        colWidths=[sidebar_w, main_w]
    )
    body.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (0, -1),  0),
        ("RIGHTPADDING",  (0, 0), (0, -1),  14),
        ("LEFTPADDING",   (1, 0), (1, -1),  14),
        ("RIGHTPADDING",  (1, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("BACKGROUND",    (0, 0), (0, -1),  SBARBG),
        ("LINEBEFORE",    (1, 0), (1, -1),  0.5, DIVIDER),
    ]))

    doc.build([body], onFirstPage=draw_header, onLaterPages=draw_header)
    buffer.seek(0)
    return buffer


def _pdf_executive(data: dict, resume: dict) -> io.BytesIO:
    """Elegant: centered header, clean uppercase section titles, single-column skills."""
    buffer = io.BytesIO()
    DARK  = colors.HexColor("#1A1A1A")
    GRAY  = colors.HexColor("#6B7280")
    LGRAY = colors.HexColor("#9CA3AF")
    SECT  = colors.HexColor("#374151")
    LM = RM = 0.8 * inch
    usable_w = letter[0] - LM - RM

    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=RM, leftMargin=LM,
        topMargin=0.8 * inch, bottomMargin=0.8 * inch
    )

    S_NAME  = ParagraphStyle("eName",  fontSize=22, fontName="Helvetica-Bold", textColor=DARK,
                              alignment=TA_CENTER, spaceAfter=4, leading=26)
    S_CON1  = ParagraphStyle("eCon1",  fontSize=10, fontName="Helvetica", textColor=GRAY,
                              alignment=TA_CENTER, spaceAfter=3, leading=13)
    S_CON2  = ParagraphStyle("eCon2",  fontSize=9,  fontName="Helvetica", textColor=LGRAY,
                              alignment=TA_CENTER, spaceAfter=12, leading=13)
    S_SEC   = ParagraphStyle("eSec",   fontSize=10.5, fontName="Helvetica-Bold", textColor=SECT,
                              spaceBefore=18, spaceAfter=7, leading=13)
    S_ETIT  = ParagraphStyle("eETit",  fontSize=11, fontName="Helvetica-Bold", textColor=DARK,
                              spaceAfter=3, leading=14)
    S_EMETA = ParagraphStyle("eEMeta", fontSize=9.5, fontName="Helvetica", textColor=GRAY,
                              spaceAfter=4, leading=13)
    S_BUL   = ParagraphStyle("eBul",   fontSize=9.5, fontName="Helvetica", textColor=DARK,
                              spaceAfter=4, leading=13, leftIndent=14, firstLineIndent=-10)
    S_OBJ   = ParagraphStyle("eObj",   fontSize=9.5, fontName="Helvetica", textColor=GRAY,
                              spaceAfter=6, leading=15)
    S_SCAT  = ParagraphStyle("eScat",  fontSize=10, fontName="Helvetica-Bold", textColor=DARK,
                              spaceAfter=3, spaceBefore=8, leading=13)
    S_SITEM = ParagraphStyle("eSitem", fontSize=9.5, fontName="Helvetica", textColor=DARK,
                              spaceAfter=2, leading=13)

    def sec(title):
        return [Paragraph(title.upper(), S_SEC)]

    def skills_single_col(skills_raw):
        skills = _normalize_skills(skills_raw)
        out = []
        for i, (cat_name, items) in enumerate(skills.items()):
            out.append(Paragraph(cat_name, S_SCAT))
            for item in (items or []):
                out.append(Paragraph(item, S_SITEM))
        return out

    contact_parts = [data.get("school", ""), f"Grade {data.get('grade', '')}"]
    if data.get("gpa"):
        contact_parts.append(f"GPA: {data['gpa']}")
    contact2_parts = []
    if data.get("email"):
        contact2_parts.append(data["email"])
    if data.get("phone"):
        contact2_parts.append(_fmt_phone(data["phone"]))

    content = []
    content.append(Paragraph(data.get("name", ""), S_NAME))
    if contact2_parts:
        content.append(Paragraph("  |  ".join(contact_parts), S_CON1))
        content.append(Paragraph("  |  ".join(contact2_parts), S_CON2))
    else:
        content.append(Paragraph("  |  ".join(contact_parts), S_CON2))
    if resume.get("objective"):
        content += sec("Summary")
        content.append(Paragraph(resume["objective"], S_OBJ))
    if resume.get("skills"):
        content += sec("Skills")
        content += skills_single_col(resume["skills"])
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


def _pdf_lateral(data: dict, resume: dict) -> io.BytesIO:
    """Chronicle: timeline main column + dark right sidebar."""
    buffer = io.BytesIO()
    PAGE_W, PAGE_H = letter
    DARK   = colors.HexColor("#1A1A1A")
    MED    = colors.HexColor("#444444")
    GRAY   = colors.HexColor("#888888")
    LGRAY  = colors.HexColor("#AAAAAA")
    SBARBG = colors.HexColor("#444444")
    SBARTX = colors.HexColor("#BBBBBB")
    SBARTITLE = colors.white
    LM = RM = 0.6 * inch
    usable_w = PAGE_W - LM - RM
    main_w   = usable_w * 0.68 - 10
    side_w   = usable_w * 0.32 - 10

    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=RM, leftMargin=LM,
        topMargin=0.65 * inch, bottomMargin=0.65 * inch
    )

    S_NAME  = ParagraphStyle("ltName", fontSize=20, fontName="Helvetica-Bold",
                              textColor=DARK, spaceAfter=3, leading=24)
    S_SUB   = ParagraphStyle("ltSub",  fontSize=10, fontName="Helvetica",
                              textColor=GRAY, spaceAfter=16, leading=14)
    S_SEC   = ParagraphStyle("ltSec",  fontSize=9,  fontName="Helvetica-Bold",
                              textColor=MED, spaceAfter=6, spaceBefore=14, leading=13,
                              borderPadding=(0, 0, 3, 0))
    S_ETIT  = ParagraphStyle("ltETit", fontSize=10.5, fontName="Helvetica-Bold",
                              textColor=DARK, spaceAfter=1, leading=14)
    S_META  = ParagraphStyle("ltMeta", fontSize=9,  fontName="Helvetica",
                              textColor=GRAY, spaceAfter=3, leading=13)
    S_DATE  = ParagraphStyle("ltDate", fontSize=8.5, fontName="Helvetica",
                              textColor=LGRAY, spaceAfter=0, leading=12)
    S_BUL   = ParagraphStyle("ltBul",  fontSize=9.5, fontName="Helvetica",
                              textColor=DARK, spaceAfter=3, leading=14,
                              leftIndent=10, firstLineIndent=-7)
    S_OBJ   = ParagraphStyle("ltObj",  fontSize=9.5, fontName="Helvetica",
                              textColor=DARK, spaceAfter=8, leading=14)
    S_STIT  = ParagraphStyle("ltSTit", fontSize=9,  fontName="Helvetica-Bold",
                              textColor=SBARTITLE, spaceAfter=6, spaceBefore=14, leading=13)
    S_SCON  = ParagraphStyle("ltSCon", fontSize=9,  fontName="Helvetica",
                              textColor=SBARTX, spaceAfter=5, leading=13)
    S_SCAT  = ParagraphStyle("ltSCat", fontSize=9,  fontName="Helvetica-Bold",
                              textColor=colors.HexColor("#DDDDDD"), spaceAfter=3,
                              spaceBefore=8, leading=13)
    S_SITEM = ParagraphStyle("ltSItm", fontSize=9,  fontName="Helvetica",
                              textColor=SBARTX, spaceAfter=3, leading=13,
                              leftIndent=8, firstLineIndent=-6)

    def _sec_rule():
        return HRFlowable(width=main_w, thickness=0.5, color=colors.HexColor("#DDDDDD"),
                          spaceAfter=8, spaceBefore=2)

    def _lateral_entries(entries):
        out = []
        for e in (entries or []):
            if isinstance(e, str):
                out.append(Paragraph(f"\u2022\u2002{e}", S_BUL))
                continue
            meta = e.get("meta", "")
            parts = meta.split(" | ") if meta else []
            date_str  = parts[-1].strip() if parts else ""
            place_str = " · ".join(p.strip() for p in parts[:-1]) if len(parts) > 1 else ""
            out.append(Paragraph(e.get("title", ""), S_ETIT))
            if date_str or place_str:
                combined = place_str
                if date_str and place_str:
                    combined = f"{place_str}   {date_str}"
                elif date_str:
                    combined = date_str
                out.append(Paragraph(combined, S_META))
            for b in e.get("bullets", []):
                out.append(Paragraph(f"\u2022\u2002{b}", S_BUL))
            out.append(Spacer(1, 8))
        return out

    # ── Main column ──
    main = []
    main.append(Paragraph(data.get("name", ""), S_NAME))
    sub_parts = [data.get("school", ""), f"Grade {data.get('grade', '')}"]
    if data.get("gpa"):
        sub_parts.append(f"GPA: {data['gpa']}")
    main.append(Paragraph("   |   ".join(sub_parts), S_SUB))

    if resume.get("objective"):
        main.append(Paragraph("SUMMARY", S_SEC))
        main.append(_sec_rule())
        main.append(Paragraph(resume["objective"], S_OBJ))
    if resume.get("experience"):
        main.append(Paragraph("EXPERIENCE", S_SEC))
        main.append(_sec_rule())
        main += _lateral_entries(resume["experience"])
    if resume.get("activities"):
        main.append(Paragraph("ACTIVITIES & LEADERSHIP", S_SEC))
        main.append(_sec_rule())
        main += _lateral_entries(resume["activities"])
    if resume.get("education"):
        main.append(Paragraph("EDUCATION", S_SEC))
        main.append(_sec_rule())
        main += _lateral_entries(resume["education"])

    # ── Sidebar ──
    side = []
    side.append(Paragraph("CONTACT", S_STIT))
    if data.get("email"):
        side.append(Paragraph(data["email"], S_SCON))
    if data.get("phone"):
        side.append(Paragraph(_fmt_phone(data["phone"]), S_SCON))
    if data.get("school"):
        side.append(Paragraph(data["school"], S_SCON))

    skills = _normalize_skills(resume.get("skills", {}))
    if skills:
        side.append(Spacer(1, 10))
        side.append(Paragraph("SKILLS", S_STIT))
        for cat_name, items in skills.items():
            side.append(Paragraph(cat_name, S_SCAT))
            for item in (items or []):
                side.append(Paragraph(f"\u2013\u2002{item}", S_SITEM))

    body = Table(
        [[main, side]],
        colWidths=[main_w, side_w]
    )
    body.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (0, -1),  0),
        ("RIGHTPADDING",  (0, 0), (0, -1),  14),
        ("LEFTPADDING",   (1, 0), (1, -1),  14),
        ("RIGHTPADDING",  (1, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("BACKGROUND",    (1, 0), (1, -1),  SBARBG),
    ]))

    doc.build([body])
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
            "email": request.form.get("email"),
            "phone": request.form.get("phone"),
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
