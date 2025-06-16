# services/generate_pdf.py
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.utils import simpleSplit
import io
from fastapi.responses import StreamingResponse


def generate_pdf_report(section_title: str, content: str) -> StreamingResponse:
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 50
    y = height - margin

    # Define layout widths (30:70 split)
    left_width = 0.3 * width
    right_width = 0.7 * width

    # === LEFT SECTION (30%) ===
    p.setFillColor(colors.lightgrey)
    p.rect(margin, margin, left_width - margin, height - 2 * margin, fill=True, stroke=False)

    # Section Title
    p.setFillColor(colors.black)
    p.setFont("Helvetica-Bold", 18)
    title_x = margin + 10
    title_y = height - margin - 40
    p.drawString(title_x, title_y, section_title)

    # === RIGHT SECTION (70%) ===
    text_x = margin + left_width + 20
    text_y = height - margin - 40
    text_width = width - text_x - margin

    # Wrap and draw content
    p.setFont("Helvetica", 11)
    wrapped_lines = simpleSplit(content.strip(), "Helvetica", 11, text_width)
    for line in wrapped_lines:
        p.drawString(text_x, text_y, line)
        text_y -= 16
        if text_y < margin + 30:
            p.showPage()
            text_y = height - margin - 40
            # Re-draw title on each new page
            p.setFillColor(colors.lightgrey)
            p.rect(margin, margin, left_width - margin, height - 2 * margin, fill=True, stroke=False)
            p.setFillColor(colors.black)
            p.setFont("Helvetica-Bold", 18)
            p.drawString(title_x, height - margin - 40, section_title)
            p.setFont("Helvetica", 11)

    p.save()
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/pdf", headers={
        "Content-Disposition": "attachment; filename=audit_report.pdf"
    })
