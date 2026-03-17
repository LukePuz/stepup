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
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import requests as http_requests

load_dotenv(override=True)

# ── Embed custom fonts ──────────────────────────────────────────────────────
def _register_pdf_fonts():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "fonts")
    _families = [
        ("Inter",             "Inter-Regular.ttf",             "Inter-Bold.ttf",             "Inter-Italic.ttf",             "Inter-BoldItalic.ttf"),
        ("OpenSans",          "OpenSans-Regular.ttf",          "OpenSans-Bold.ttf",          "OpenSans-Italic.ttf",          "OpenSans-BoldItalic.ttf"),
        ("EBGaramond",        "EBGaramond-Regular.ttf",        "EBGaramond-Bold.ttf",        "EBGaramond-Italic.ttf",        None),
        ("DMSans",            "DMSans-Regular.ttf",            "DMSans-Bold.ttf",            "DMSans-Italic.ttf",            None),
        ("CormorantGaramond", "CormorantGaramond-Regular.ttf", "CormorantGaramond-Bold.ttf", "CormorantGaramond-Italic.ttf", None),
        ("Jost",              "Jost-Regular.ttf",              "Jost-Bold.ttf",              "Jost-Italic.ttf",              None),
    ]
    for name, regular, bold, italic, bold_italic in _families:
        try:
            pdfmetrics.registerFont(TTFont(name,                  os.path.join(base, regular)))
            pdfmetrics.registerFont(TTFont(f"{name}-Bold",        os.path.join(base, bold)))
            pdfmetrics.registerFont(TTFont(f"{name}-Italic",      os.path.join(base, italic)))
            bi = bold_italic or italic
            pdfmetrics.registerFont(TTFont(f"{name}-BoldItalic",  os.path.join(base, bi)))
            pdfmetrics.registerFontFamily(name, normal=name, bold=f"{name}-Bold",
                                          italic=f"{name}-Italic", boldItalic=f"{name}-BoldItalic")
        except Exception as _e:
            print(f"[fonts] Warning: could not register {name}: {_e}")

_register_pdf_fonts()

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
        response_format={"type": "json_object"},
        timeout=90
    )

    return json.loads(response.choices[0].message.content)


def generate_cover_letter(data: dict, resume: dict) -> str:
    activities_text = ""
    for a in resume.get("activities", []):
        if isinstance(a, dict):
            bullets = "; ".join(a.get("bullets", []))
            activities_text += f"- {a.get('title', '')} ({a.get('meta', '')}): {bullets}\n"

    experience_text = ""
    for e in resume.get("experience", []):
        if isinstance(e, dict):
            bullets = "; ".join(e.get("bullets", []))
            experience_text += f"- {e.get('title', '')} ({e.get('meta', '')}): {bullets}\n"

    prompt = f"""Write a cover letter body for a high school student applying for: {data.get('applying_for')}.

Student details:
- Name: {data.get('name')}
- School: {data.get('school')}, Grade {data.get('grade')}
- Activities:
{activities_text or 'None'}
- Experience:
{experience_text or 'None'}

Requirements:
- Exactly 3 paragraphs, ~180 words total
- Paragraph 1: Jump straight into why this specific opportunity interests them — no hollow openers like "I am writing to express my interest"
- Paragraph 2: Reference 2 things from their background using only details explicitly listed above — titles, roles, and bullets as written. Do not add outcomes, statistics, or stories that are not in the provided data.
- Paragraph 3: One confident closing sentence, then a brief thank-you

Fabrication rules — this is the most important section:
- Do NOT invent any detail that is not explicitly stated in the student's activities or experience above
- Do NOT add scores, rankings, outcomes, or results (e.g. "went from last place to the playoffs") unless they appear word-for-word in the bullets
- Do NOT add context about the organization (e.g. "I've followed NASA's missions closely") unless the student mentioned it
- If the provided bullets are sparse, write shorter, vaguer sentences — do not fill the gap with invented specifics
- Every factual claim must be traceable to the data above

Tone rules:
- Use short, direct sentences. Vary the length.
- No buzzwords: do not use "passionate", "leverage", "eager to contribute", "hard-working", "dedicated", "honed my skills", "I would be a great fit", "I am confident that"
- No filler transitions like "Furthermore," or "In conclusion,"
- Professional but natural — the way a smart student actually talks, not corporate-speak

Do not include a date, address, salutation, or sign-off. Return only the 3 paragraph body."""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You write cover letters that sound like real, articulate high school students — professional but natural, specific not generic. Return only the letter body."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def build_pdf(data: dict, resume: dict, template: str = "classic") -> io.BytesIO:
    if template == "modern":
        return _pdf_modern(data, resume)
    elif template == "executive":
        return _pdf_executive(data, resume)
    elif template == "lateral":
        return _pdf_lateral(data, resume)
    elif template == "lumina":
        return _pdf_lumina(data, resume)
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
    """Professional: warm sidebar, serif name on canvas, full-width name divider."""
    buffer = io.BytesIO()
    PAGE_W, PAGE_H = letter

    # ── Palette ──────────────────────────────────────────────────────────────
    BLACK    = colors.HexColor("#111111")
    DARK     = colors.HexColor("#1A1A1A")
    MID      = colors.HexColor("#4D4D4D")
    MUTED    = colors.HexColor("#888888")
    RULE     = colors.HexColor("#C9C6C2")
    SBARBG   = colors.HexColor("#F4F2EF")
    SBAREDGE = colors.HexColor("#D8D5D1")

    # ── Layout ───────────────────────────────────────────────────────────────
    LM = RM = 0.55 * inch
    usable_w = PAGE_W - LM - RM
    side_w   = usable_w * 0.305
    main_w   = usable_w - side_w

    applying = (data.get("applying_for") or "").strip()
    HDR_H    = (1.38 if applying else 1.14) * inch

    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=RM, leftMargin=LM,
        topMargin=HDR_H, bottomMargin=0.55 * inch
    )

    # ── Styles ───────────────────────────────────────────────────────────────
    S_SLBL  = ParagraphStyle("pfSLbl",  fontSize=7,   fontName="Jost-Bold",
                              textColor=DARK,  spaceAfter=5, spaceBefore=18, leading=9)
    S_SLBL_F= ParagraphStyle("pfSLblF", fontSize=7,   fontName="Jost-Bold",
                              textColor=DARK,  spaceAfter=5, leading=9)
    S_CITEM = ParagraphStyle("pfCItm",  fontSize=9,   fontName="Jost",
                              textColor=MID,   spaceAfter=5, leading=13)
    S_ESCH  = ParagraphStyle("pfESch",  fontSize=10,  fontName="Jost-Bold",
                              textColor=DARK,  spaceAfter=2, leading=14)
    S_EGRD  = ParagraphStyle("pfEGrd",  fontSize=9,   fontName="Jost",
                              textColor=MID,   spaceAfter=3, leading=13)
    S_EGPA  = ParagraphStyle("pfEGpa",  fontSize=9,   fontName="Jost-Bold",
                              textColor=DARK,  spaceAfter=0, leading=12)
    S_SGNAME= ParagraphStyle("pfSGNm",  fontSize=8.5, fontName="Jost-Bold",
                              textColor=DARK,  spaceAfter=4, spaceBefore=14, leading=12)
    S_SGITM = ParagraphStyle("pfSGItm", fontSize=9,   fontName="Jost",
                              textColor=MID,   spaceAfter=3, leading=13, leftIndent=8)
    S_MLBL  = ParagraphStyle("pfMLbl",  fontSize=7,   fontName="Jost-Bold",
                              textColor=DARK,  spaceAfter=6, spaceBefore=20, leading=9)
    S_MLBL_F= ParagraphStyle("pfMLblF", fontSize=7,   fontName="Jost-Bold",
                              textColor=DARK,  spaceAfter=6, leading=9)
    S_SUM   = ParagraphStyle("pfSum",   fontSize=9.5, fontName="Jost",
                              textColor=MID,   spaceAfter=0, leading=16)
    S_ETIT  = ParagraphStyle("pfETit",  fontSize=11,  fontName="Jost-Bold",
                              textColor=DARK,  spaceAfter=1, leading=14)
    S_EDATE = ParagraphStyle("pfEDt",   fontSize=8.5, fontName="Jost",
                              textColor=MUTED, alignment=2, leading=14)
    S_ESUB  = ParagraphStyle("pfESub",  fontSize=9,   fontName="Jost-Italic",
                              textColor=MID,   spaceAfter=5, leading=13)
    S_BUL   = ParagraphStyle("pfBul",   fontSize=9.5, fontName="Jost",
                              textColor=MID,   spaceAfter=4, leading=15,
                              leftIndent=13, firstLineIndent=-11)
    S_ADDL  = ParagraphStyle("pfAddl",  fontSize=9.5, fontName="Jost",
                              textColor=MID,   spaceAfter=0, leading=16)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _sb_rule():
        return HRFlowable(width=side_w, thickness=0.5, color=RULE, spaceAfter=7, spaceBefore=0)

    def _main_rule():
        return HRFlowable(width=main_w, thickness=0.5, color=RULE, spaceAfter=8, spaceBefore=0)

    def _sb_label(text, first=False):
        s = S_SLBL_F if first else S_SLBL
        return [Paragraph(text.upper(), s), _sb_rule()]

    def _main_label(text, first=False):
        s = S_MLBL_F if first else S_MLBL
        return [Paragraph(text.upper(), s), _main_rule()]

    DATE_COL = 0.85 * inch

    def _pf_entries(entries):
        out = []
        for e in (entries or []):
            if isinstance(e, str):
                out.append(Paragraph(f"\u2013\u2002{e}", S_BUL))
                continue
            meta  = e.get("meta", "")
            parts = meta.split(" | ") if meta else []
            dur   = parts[-1].strip() if parts else ""
            sub   = " \u00b7 ".join(p.strip() for p in parts[:-1]) if len(parts) > 1 else ""
            title_w = main_w - DATE_COL
            if dur:
                row = Table([[Paragraph(e.get("title", ""), S_ETIT), Paragraph(dur, S_EDATE)]],
                            colWidths=[title_w, DATE_COL])
                row.setStyle(TableStyle([
                    ("VALIGN",        (0, 0), (-1, -1), "BOTTOM"),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
                    ("TOPPADDING",    (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]))
                out.append(row)
            else:
                out.append(Paragraph(e.get("title", ""), S_ETIT))
            if sub:
                out.append(Paragraph(sub, S_ESUB))
            for b in e.get("bullets", []):
                out.append(Paragraph(f"\u2013\u2002{b}", S_BUL))
            out.append(Spacer(1, 10))
        return out

    # ── Sidebar ───────────────────────────────────────────────────────────────
    side = []
    side += _sb_label("Contact", first=True)
    if data.get("email"):
        side.append(Paragraph(data["email"], S_CITEM))
    if data.get("phone"):
        side.append(Paragraph(_fmt_phone(data["phone"]), S_CITEM))
    side += _sb_label("Education")
    side.append(Paragraph(data.get("school", ""), S_ESCH))
    side.append(Paragraph(f"Grade {data.get('grade', '')}  \u00b7  Current", S_EGRD))
    if data.get("gpa"):
        side.append(Paragraph(f"GPA: {data['gpa']}", S_EGPA))
    skills = _normalize_skills(resume.get("skills", {}))
    if skills:
        side += _sb_label("Skills")
        for cat_name, items in skills.items():
            side.append(Paragraph(cat_name, S_SGNAME))
            for item in (items or []):
                side.append(Paragraph(item, S_SGITM))

    # ── Main sections (name drawn on canvas) ──────────────────────────────────
    main = []
    first_sec = True
    if resume.get("objective"):
        main += _main_label("Summary", first=first_sec); first_sec = False
        main.append(Paragraph(resume["objective"], S_SUM))
    if resume.get("activities"):
        main += _main_label("Activities & Involvement", first=first_sec); first_sec = False
        main += _pf_entries(resume["activities"])
    if resume.get("experience"):
        main += _main_label("Work & Volunteer Experience", first=first_sec); first_sec = False
        main += _pf_entries(resume["experience"])
    if data.get("extra"):
        main += _main_label("Additional Information", first=first_sec)
        main.append(Paragraph(data["extra"], S_ADDL))

    # ── Canvas: sidebar bg + name header + full-width rule ────────────────────
    name_x   = LM + side_w + 18          # aligned with main content left edge
    name_y   = PAGE_H - 0.60 * inch      # name baseline
    tagl_y   = name_y - 0.40 * inch      # tagline below name
    appl_y   = tagl_y - 0.18 * inch      # applying-for (optional)
    hr_y     = PAGE_H - HDR_H + 0.10 * inch

    def draw_page(canvas, doc):
        canvas.saveState()
        # Sidebar warm background, flush to left edge
        canvas.setFillColor(SBARBG)
        canvas.rect(0, 0, LM + side_w, PAGE_H, fill=1, stroke=0)
        # Sidebar right border
        canvas.setStrokeColor(SBAREDGE)
        canvas.setLineWidth(0.75)
        canvas.line(LM + side_w, 0, LM + side_w, PAGE_H)
        # Name
        canvas.setFillColor(BLACK)
        canvas.setFont("CormorantGaramond-Bold", 30)
        canvas.drawString(name_x, name_y, data.get("name", ""))
        # Tagline
        canvas.setFont("Jost", 7.5)
        canvas.setFillColor(MUTED)
        tagline = f"{data.get('school', '').upper()}   \u00b7   GRADE {data.get('grade', '')}"
        canvas.drawString(name_x, tagl_y, tagline)
        if applying:
            canvas.drawString(name_x, appl_y, applying.upper())
        # Full-width rule (left margin to right margin)
        canvas.setStrokeColor(BLACK)
        canvas.setLineWidth(1.2)
        canvas.line(LM, hr_y, PAGE_W - RM, hr_y)
        canvas.restoreState()

    # ── Assemble ──────────────────────────────────────────────────────────────
    body = Table([[side, main]], colWidths=[side_w, main_w])
    body.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (0, -1),  0),
        ("RIGHTPADDING",  (0, 0), (0, -1),  16),
        ("LEFTPADDING",   (1, 0), (1, -1),  18),
        ("RIGHTPADDING",  (1, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    doc.build([body], onFirstPage=draw_page, onLaterPages=draw_page)
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
        canvas.setFont("Inter-Bold", 24)
        canvas.drawCentredString(PAGE_W / 2, PAGE_H - 0.46 * inch, name_text)
        canvas.setFillColor(LGRAY)
        canvas.setFont("OpenSans", 9)
        canvas.drawCentredString(PAGE_W / 2, PAGE_H - 0.72 * inch, contact_text)
        if contact2_text:
            canvas.drawCentredString(PAGE_W / 2, PAGE_H - 0.91 * inch, contact2_text)
        canvas.restoreState()

    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=RM, leftMargin=LM,
        topMargin=HEADER_H + 0.20 * inch, bottomMargin=0.68 * inch
    )

    S_ETIT  = ParagraphStyle("mETit",  fontSize=11,   fontName="Inter-Bold",  textColor=NAVY,
                              spaceAfter=2,  leading=14)
    S_EDATE = ParagraphStyle("mEDate", fontSize=9,    fontName="OpenSans",    textColor=GRAY,
                              alignment=2,  leading=14)   # TA_RIGHT = 2
    S_EMETA = ParagraphStyle("mEMeta", fontSize=9.5,  fontName="OpenSans",    textColor=GRAY,
                              spaceAfter=4, leading=13)
    S_BUL   = ParagraphStyle("mBul",   fontSize=10,   fontName="OpenSans",    textColor=DARK,
                              spaceAfter=5, leading=15, leftIndent=13, firstLineIndent=-9)
    S_OBJ   = ParagraphStyle("mObj",   fontSize=10,   fontName="OpenSans",    textColor=DARK,
                              spaceAfter=8, leading=16)
    S_SCAT  = ParagraphStyle("mScat",  fontSize=9.5,  fontName="Inter-Bold",  textColor=DARK,
                              spaceAfter=3, spaceBefore=10, leading=13)
    S_SITEM = ParagraphStyle("mSitem", fontSize=9.5,  fontName="OpenSans",    textColor=DARK,
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
            c.setFont("Inter-Bold", 9)
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

    S_NAME  = ParagraphStyle("eName",  fontSize=22, fontName="Inter-Bold", textColor=DARK,
                              alignment=TA_CENTER, spaceAfter=4, leading=26)
    S_CON1  = ParagraphStyle("eCon1",  fontSize=10, fontName="OpenSans", textColor=GRAY,
                              alignment=TA_CENTER, spaceAfter=3, leading=13)
    S_CON2  = ParagraphStyle("eCon2",  fontSize=9,  fontName="OpenSans", textColor=LGRAY,
                              alignment=TA_CENTER, spaceAfter=12, leading=13)
    S_SEC   = ParagraphStyle("eSec",   fontSize=10.5, fontName="Inter-Bold", textColor=SECT,
                              spaceBefore=18, spaceAfter=7, leading=13)
    S_ETIT  = ParagraphStyle("eETit",  fontSize=11, fontName="Inter-Bold", textColor=DARK,
                              spaceAfter=3, leading=14)
    S_EMETA = ParagraphStyle("eEMeta", fontSize=9.5, fontName="OpenSans", textColor=GRAY,
                              spaceAfter=4, leading=13)
    S_BUL   = ParagraphStyle("eBul",   fontSize=9.5, fontName="OpenSans", textColor=DARK,
                              spaceAfter=4, leading=13, leftIndent=14, firstLineIndent=-10)
    S_OBJ   = ParagraphStyle("eObj",   fontSize=9.5, fontName="OpenSans", textColor=GRAY,
                              spaceAfter=6, leading=15)
    S_SCAT  = ParagraphStyle("eScat",  fontSize=10, fontName="Inter-Bold", textColor=DARK,
                              spaceAfter=3, spaceBefore=8, leading=13)
    S_SITEM = ParagraphStyle("eSitem", fontSize=9.5, fontName="OpenSans", textColor=DARK,
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

    S_NAME  = ParagraphStyle("ltName", fontSize=20, fontName="Inter-Bold",
                              textColor=DARK, spaceAfter=3, leading=24)
    S_SUB   = ParagraphStyle("ltSub",  fontSize=10, fontName="OpenSans",
                              textColor=GRAY, spaceAfter=16, leading=14)
    S_SEC   = ParagraphStyle("ltSec",  fontSize=9,  fontName="Inter-Bold",
                              textColor=MED, spaceAfter=6, spaceBefore=14, leading=13,
                              borderPadding=(0, 0, 3, 0))
    S_ETIT  = ParagraphStyle("ltETit", fontSize=10.5, fontName="Inter-Bold",
                              textColor=DARK, spaceAfter=1, leading=14)
    S_META  = ParagraphStyle("ltMeta", fontSize=9,  fontName="OpenSans",
                              textColor=GRAY, spaceAfter=3, leading=13)
    S_DATE  = ParagraphStyle("ltDate", fontSize=8.5, fontName="OpenSans",
                              textColor=LGRAY, spaceAfter=0, leading=12)
    S_BUL   = ParagraphStyle("ltBul",  fontSize=9.5, fontName="OpenSans",
                              textColor=DARK, spaceAfter=3, leading=14,
                              leftIndent=10, firstLineIndent=-7)
    S_OBJ   = ParagraphStyle("ltObj",  fontSize=9.5, fontName="OpenSans",
                              textColor=DARK, spaceAfter=8, leading=14)
    S_STIT  = ParagraphStyle("ltSTit", fontSize=9,  fontName="Inter-Bold",
                              textColor=SBARTITLE, spaceAfter=6, spaceBefore=14, leading=13)
    S_SCON  = ParagraphStyle("ltSCon", fontSize=9,  fontName="OpenSans",
                              textColor=SBARTX, spaceAfter=5, leading=13)
    S_SCAT  = ParagraphStyle("ltSCat", fontSize=9,  fontName="Inter-Bold",
                              textColor=colors.HexColor("#DDDDDD"), spaceAfter=3,
                              spaceBefore=8, leading=13)
    S_SITEM = ParagraphStyle("ltSItm", fontSize=9,  fontName="OpenSans",
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


def _pdf_lumina(data: dict, resume: dict) -> io.BytesIO:
    """Lumina: serif name, navy header, light sidebar, blue accents."""
    from reportlab.platypus import Flowable as _Flowable

    buffer = io.BytesIO()
    PAGE_W, PAGE_H = letter
    NAVY    = colors.HexColor("#1B2A45")
    ACCENT  = colors.HexColor("#2E5FA3")
    ACCL    = colors.HexColor("#EAF0FA")
    DARK    = colors.HexColor("#1A1A1A")
    MEDGRAY = colors.HexColor("#4A4A4A")
    GRAY    = colors.HexColor("#888888")
    SBARBG  = colors.HexColor("#F4F6FA")
    DIVIDER = colors.HexColor("#D4D9E2")
    LM = RM = 0.65 * inch
    usable_w = PAGE_W - LM - RM
    side_w   = usable_w * 0.30 - 8
    main_w   = usable_w * 0.70 - 8

    applying = (data.get("applying_for") or "").strip()
    HEADER_H = (1.65 if applying else 1.4) * inch

    name_text    = data.get("name", "")
    school_text  = data.get("school", "")
    grade_text   = f"Grade {data.get('grade', '')}"
    gpa_text     = f"GPA {data['gpa']}" if data.get("gpa") else ""
    meta_right   = "  ·  ".join(filter(None, [grade_text, gpa_text]))
    email_text   = data.get("email", "")
    phone_text   = _fmt_phone(data.get("phone", "") or "")

    def draw_header(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, PAGE_H - HEADER_H, PAGE_W, HEADER_H, fill=1, stroke=0)
        # Name (large serif)
        canvas.setFillColor(colors.white)
        canvas.setFont("EBGaramond-Bold", 26)
        canvas.drawString(LM, PAGE_H - 0.52 * inch, name_text)
        # School + grade on right
        canvas.setFont("DMSans-Bold", 11)
        canvas.setFillColor(colors.HexColor("#E0E6F0"))
        canvas.drawRightString(PAGE_W - RM, PAGE_H - 0.47 * inch, school_text)
        canvas.setFont("DMSans", 10)
        canvas.setFillColor(colors.HexColor("#94A9C4"))
        canvas.drawRightString(PAGE_W - RM, PAGE_H - 0.63 * inch, meta_right)
        # Applying-for strip
        y_after_name = PAGE_H - 0.75 * inch
        if applying:
            canvas.setFillColor(colors.HexColor("#263D60"))
            canvas.roundRect(LM, y_after_name - 0.18 * inch, 2.8 * inch, 0.20 * inch, 3, fill=1, stroke=0)
            canvas.setFillColor(colors.HexColor("#94A9C4"))
            canvas.setFont("DMSans", 8.5)
            canvas.drawString(LM + 9, y_after_name - 0.13 * inch, "Applying for: ")
            canvas.setFillColor(colors.white)
            canvas.setFont("DMSans-Bold", 8.5)
            canvas.drawString(LM + 72, y_after_name - 0.13 * inch, applying)
            y_after_name -= 0.24 * inch
        # Contact bar separator
        sep_y = y_after_name - 0.08 * inch
        canvas.setStrokeColor(colors.HexColor("#2E3F5E"))
        canvas.setLineWidth(0.5)
        canvas.line(LM, sep_y, PAGE_W - RM, sep_y)
        # Email + phone
        canvas.setFont("DMSans", 9.5)
        canvas.setFillColor(colors.HexColor("#94A9C4"))
        contact_y = sep_y - 0.20 * inch
        x = LM
        if email_text:
            canvas.drawString(x, contact_y, email_text)
            x += canvas.stringWidth(email_text, "DMSans", 9.5) + 24
        if phone_text:
            canvas.drawString(x, contact_y, phone_text)
        canvas.restoreState()

    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=RM, leftMargin=LM,
        topMargin=HEADER_H + 0.22 * inch, bottomMargin=0.65 * inch
    )

    # Styles
    S_SBLABEL = ParagraphStyle("lmSBLbl", fontSize=8, fontName="DMSans-Bold",
                                textColor=ACCENT, spaceAfter=6, spaceBefore=20,
                                leading=11, wordWrap='LTR')
    S_SBLABEL_FIRST = ParagraphStyle("lmSBLbl1", fontSize=8, fontName="DMSans-Bold",
                                      textColor=ACCENT, spaceAfter=6,
                                      leading=11, wordWrap='LTR')
    S_SCHOOL  = ParagraphStyle("lmSch",  fontSize=11, fontName="DMSans-Bold",
                                textColor=DARK, spaceAfter=2, leading=14)
    S_EDUMETA = ParagraphStyle("lmEduM", fontSize=9.5, fontName="DMSans",
                                textColor=GRAY, spaceAfter=5, leading=13)
    S_GPABADGE= ParagraphStyle("lmGPA",  fontSize=9,  fontName="DMSans-Bold",
                                textColor=colors.white, spaceAfter=0, leading=12,
                                backColor=NAVY, borderPadding=(2, 6, 2, 6))
    S_SGCAT   = ParagraphStyle("lmSGCat",fontSize=9,  fontName="DMSans-Bold",
                                textColor=DARK, spaceAfter=3, spaceBefore=8, leading=13)
    S_SGITEM  = ParagraphStyle("lmSGItm",fontSize=9.5, fontName="DMSans",
                                textColor=MEDGRAY, spaceAfter=2, leading=13,
                                leftIndent=8, firstLineIndent=-6)
    S_SECTIT  = ParagraphStyle("lmSecT", fontSize=8,  fontName="DMSans-Bold",
                                textColor=ACCENT, spaceAfter=8, spaceBefore=20,
                                leading=11, wordWrap='LTR')
    S_SECTIT_FIRST = ParagraphStyle("lmSecT1", fontSize=8, fontName="DMSans-Bold",
                                     textColor=ACCENT, spaceAfter=8,
                                     leading=11, wordWrap='LTR')
    S_SUMMARY = ParagraphStyle("lmSum",  fontSize=10.5, fontName="DMSans",
                                textColor=MEDGRAY, spaceAfter=0, leading=17)
    S_ETIT    = ParagraphStyle("lmETit", fontSize=12,  fontName="DMSans-Bold",
                                textColor=DARK, spaceAfter=1, leading=15)
    S_EDATE   = ParagraphStyle("lmEDt",  fontSize=9.5, fontName="DMSans",
                                textColor=GRAY, alignment=2, leading=15)
    S_ESUB    = ParagraphStyle("lmESub", fontSize=10,  fontName="DMSans",
                                textColor=ACCENT, spaceAfter=5, leading=13)
    S_BUL     = ParagraphStyle("lmBul",  fontSize=10,  fontName="DMSans",
                                textColor=MEDGRAY, spaceAfter=3, leading=15,
                                leftIndent=12, firstLineIndent=-9)

    class SectionTitle(_Flowable):
        """Blue left-bar accent + uppercase label, similar to Lumina design."""
        def __init__(self, text, col_w, first=False):
            _Flowable.__init__(self)
            self.text  = text.upper()
            self.col_w = col_w
            self._pad  = 0 if first else 18
            self._h    = 22

        def wrap(self, aw, ah):
            return (self.col_w, self._h + self._pad)

        def draw(self):
            c = self.canv
            c.saveState()
            c.translate(0, self._pad)
            # Blue left bar
            c.setFillColor(ACCENT)
            c.rect(0, 4, 3, 12, fill=1, stroke=0)
            # Text
            c.setFont("DMSans-Bold", 8)
            c.setFillColor(ACCENT)
            c.drawString(10, 7, self.text)
            # Rule below
            c.setStrokeColor(ACCL)
            c.setLineWidth(1.2)
            c.line(0, 2, self.col_w, 2)
            c.restoreState()

    def _sb_section_title(text, first=False):
        s = S_SBLABEL_FIRST if first else S_SBLABEL
        out = [Paragraph(text.upper(), s)]
        out.append(HRFlowable(width=side_w, thickness=1.2, color=ACCL, spaceAfter=6, spaceBefore=0))
        return out

    DATE_COL = 0.85 * inch

    def _lm_entries(entries, first_sec=False):
        out = []
        entries = entries if isinstance(entries, list) else []
        for i, e in enumerate(entries):
            if isinstance(e, str):
                out.append(Paragraph(f"\u2022\u2002{e}", S_BUL))
                continue
            meta   = e.get("meta", "")
            parts  = meta.split(" | ") if meta else []
            dur    = parts[-1].strip() if parts else ""
            sub    = " · ".join(p.strip() for p in parts[:-1]) if len(parts) > 1 else ""
            title_w = main_w - DATE_COL
            if dur:
                row = Table([[Paragraph(e.get("title", ""), S_ETIT), Paragraph(dur, S_EDATE)]],
                            colWidths=[title_w, DATE_COL])
                row.setStyle(TableStyle([
                    ("VALIGN",        (0,0),(-1,-1),"BOTTOM"),
                    ("LEFTPADDING",   (0,0),(-1,-1),0),
                    ("RIGHTPADDING",  (0,0),(-1,-1),0),
                    ("TOPPADDING",    (0,0),(-1,-1),0),
                    ("BOTTOMPADDING", (0,0),(-1,-1),2),
                ]))
                out.append(row)
            else:
                out.append(Paragraph(e.get("title", ""), S_ETIT))
            if sub:
                out.append(Paragraph(sub, S_ESUB))
            for b in e.get("bullets", []):
                out.append(Paragraph(f"\u2022\u2002{b}", S_BUL))
            if i < len(entries) - 1:
                out.append(HRFlowable(width=main_w, thickness=0.5, color=colors.HexColor("#ECEEF2"),
                                      spaceAfter=10, spaceBefore=10))
        return out

    # ── Sidebar ──
    side = []
    side += _sb_section_title("Education", first=True)
    side.append(Paragraph(data.get("school", ""), S_SCHOOL))
    grade_meta = f"Grade {data.get('grade', '')}  ·  Current"
    side.append(Paragraph(grade_meta, S_EDUMETA))
    if data.get("gpa"):
        side.append(Paragraph(f"GPA: {data['gpa']}", S_GPABADGE))

    skills = _normalize_skills(resume.get("skills", {}))
    if skills:
        side += _sb_section_title("Skills")
        for cat_name, items in skills.items():
            side.append(Paragraph(cat_name, S_SGCAT))
            for item in (items or []):
                side.append(Paragraph(f"\u2013\u2002{item}", S_SGITEM))

    # ── Main ──
    main = []
    first = True
    if resume.get("objective"):
        main.append(SectionTitle("Summary", main_w, first=first)); first = False
        main.append(Paragraph(resume["objective"], S_SUMMARY))
    if resume.get("activities"):
        main.append(SectionTitle("Activities & Involvement", main_w, first=first)); first = False
        main += _lm_entries(resume["activities"])
    if resume.get("experience"):
        main.append(SectionTitle("Work & Volunteer Experience", main_w, first=first)); first = False
        main += _lm_entries(resume["experience"])
    if data.get("extra"):
        main.append(SectionTitle("Additional Information", main_w, first=first))
        main.append(Paragraph(data["extra"], S_SUMMARY))

    body = Table([[side, main]], colWidths=[side_w, main_w])
    body.setStyle(TableStyle([
        ("VALIGN",        (0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",   (0,0),(0,-1), 0),
        ("RIGHTPADDING",  (0,0),(0,-1), 14),
        ("LEFTPADDING",   (1,0),(1,-1), 16),
        ("RIGHTPADDING",  (1,0),(-1,-1),0),
        ("TOPPADDING",    (0,0),(-1,-1),0),
        ("BOTTOMPADDING", (0,0),(-1,-1),0),
        ("BACKGROUND",    (0,0),(0,-1), SBARBG),
        ("LINEBEFORE",    (1,0),(1,-1), 0.5, DIVIDER),
    ]))

    doc.build([body], onFirstPage=draw_header, onLaterPages=draw_header)
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
        try:
            resume = generate_resume(data)
        except Exception:
            return render_template("build.html", prefill=data,
                error="Something went wrong while generating your resume. Please try again in a moment.")
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
    try:
        pdf = build_pdf(data, resume, template)
    except Exception:
        return "PDF generation failed. Please go back and try again.", 500
    return send_file(
        pdf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"{data.get('name', 'resume').replace(' ', '_')}_resume.pdf"
    )


@app.route("/refine", methods=["POST"])
def refine():
    data = json.loads(request.form.get("data"))
    resume = json.loads(request.form.get("resume"))
    return render_template("result.html", data=data, resume=resume)


@app.route("/cover-letter", methods=["POST"])
def cover_letter():
    data = json.loads(request.form.get("data"))
    resume = json.loads(request.form.get("resume"))
    try:
        letter = generate_cover_letter(data, resume)
    except Exception:
        letter = None
    try:
        http_requests.post(
            f"{os.getenv('SUPABASE_URL')}/rest/v1/resume-generations",
            json={"template": "cover-letter"},
            headers={
                "apikey": os.getenv("SUPABASE_KEY"),
                "Authorization": f"Bearer {os.getenv('SUPABASE_KEY')}",
                "Content-Type": "application/json"
            }
        )
    except Exception:
        pass
    return render_template("result.html", data=data, resume=resume, cover_letter=letter)


@app.errorhandler(429)
def rate_limit_exceeded(e):
    return render_template("limit.html"), 429


if __name__ == "__main__":
    app.run(debug=True)
