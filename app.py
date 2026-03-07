import os
import json
import io
from flask import Flask, render_template, request, send_file
from openai import OpenAI
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib import colors
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

load_dotenv(override=True)

app = Flask(__name__)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://"
)


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


def build_pdf(data: dict, resume: dict) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch
    )

    styles = getSampleStyleSheet()

    name_style = ParagraphStyle("Name", fontSize=18, fontName="Helvetica-Bold", textColor=colors.HexColor("#111111"), spaceAfter=6, leading=22)
    sub_style = ParagraphStyle("Sub", fontSize=10, fontName="Helvetica", textColor=colors.HexColor("#555555"), spaceAfter=10, leading=14)
    section_style = ParagraphStyle("Section", fontSize=11, fontName="Helvetica-Bold", textColor=colors.HexColor("#6366f1"), spaceBefore=12, spaceAfter=4)
    body_style = ParagraphStyle("Body", fontSize=10, fontName="Helvetica", textColor=colors.HexColor("#222222"), spaceAfter=3, leftIndent=12)
    objective_style = ParagraphStyle("Obj", fontSize=10, fontName="Helvetica-Oblique", textColor=colors.HexColor("#444444"), spaceAfter=8)

    content = []

    # Header
    content.append(Paragraph(data.get("name", ""), name_style))
    content.append(Paragraph(f"{data.get('school', '')} &bull; Grade {data.get('grade', '')} &bull; GPA: {data.get('gpa', '')}", sub_style))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e5e7eb")))

    # Objective
    content.append(Paragraph("OBJECTIVE", section_style))
    content.append(Paragraph(resume.get("objective", ""), objective_style))

    # Education
    content.append(Paragraph("EDUCATION", section_style))
    content.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb")))
    content.append(Spacer(1, 4))
    for bullet in resume.get("education", []):
        content.append(Paragraph(f"• {bullet}", body_style))

    # Activities
    if resume.get("activities"):
        content.append(Paragraph("ACTIVITIES & LEADERSHIP", section_style))
        content.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb")))
        content.append(Spacer(1, 4))
        for bullet in resume.get("activities", []):
            content.append(Paragraph(f"• {bullet}", body_style))

    # Experience
    if resume.get("experience"):
        content.append(Paragraph("EXPERIENCE", section_style))
        content.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb")))
        content.append(Spacer(1, 4))
        for bullet in resume.get("experience", []):
            content.append(Paragraph(f"• {bullet}", body_style))

    # Skills
    if resume.get("skills"):
        content.append(Paragraph("SKILLS", section_style))
        content.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb")))
        content.append(Spacer(1, 4))
        content.append(Paragraph(", ".join(resume.get("skills", [])), body_style))

    doc.build(content)
    buffer.seek(0)
    return buffer


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/build", methods=["GET", "POST"])
@limiter.limit("3 per day", methods=["POST"])
def build():
    if request.method == "POST":
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
    return render_template("build.html")


@app.route("/download", methods=["POST"])
def download():
    data = json.loads(request.form.get("data"))
    resume = json.loads(request.form.get("resume"))
    pdf = build_pdf(data, resume)
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
