# services/generate_pdf.py
# from reportlab.pdfgen import canvas
# from reportlab.lib.pagesizes import A4
# from reportlab.lib import colors
# from reportlab.lib.utils import simpleSplit
# import io
# from fastapi.responses import StreamingResponse


# def generate_pdf_report(section_title: str, content: str) -> StreamingResponse:
#     """Generate a PDF report with the given title and content"""
#     try:
#         print(f"📄 Generating PDF with title: {section_title}")
        
#         if not content or not content.strip():
#             content = "No content available for this report."
        
#         buffer = io.BytesIO()
#         p = canvas.Canvas(buffer, pagesize=A4)
#         width, height = A4
#         margin = 50
#         y = height - margin

#         # Define layout widths (30:70 split)
#         left_width = 0.3 * width
#         right_width = 0.7 * width

#         # === LEFT SECTION (30%) ===
#         p.setFillColor(colors.lightgrey)
#         p.rect(margin, margin, left_width - margin, height - 2 * margin, fill=True, stroke=False)

#         # Section Title
#         p.setFillColor(colors.black)
#         p.setFont("Helvetica-Bold", 18)
#         title_x = margin + 10
#         title_y = height - margin - 40
        
#         # Handle long titles by wrapping them
#         title_lines = simpleSplit(section_title, "Helvetica-Bold", 18, left_width - 30)
#         for i, title_line in enumerate(title_lines):
#             p.drawString(title_x, title_y - (i * 22), title_line)

#         # === RIGHT SECTION (70%) ===
#         text_x = margin + left_width + 20
#         text_y = height - margin - 40
#         text_width = width - text_x - margin

#         # Wrap and draw content
#         p.setFont("Helvetica", 11)
#         content_lines = content.strip().split('\n')
        
#         for paragraph in content_lines:
#             if paragraph.strip():  # Skip empty lines
#                 wrapped_lines = simpleSplit(paragraph.strip(), "Helvetica", 11, text_width)
#                 for line in wrapped_lines:
#                     if text_y < margin + 30:
#                         p.showPage()
#                         text_y = height - margin - 40
#                         # Re-draw left section on each new page
#                         p.setFillColor(colors.lightgrey)
#                         p.rect(margin, margin, left_width - margin, height - 2 * margin, fill=True, stroke=False)
#                         p.setFillColor(colors.black)
#                         p.setFont("Helvetica-Bold", 18)
#                         for i, title_line in enumerate(title_lines):
#                             p.drawString(title_x, height - margin - 40 - (i * 22), title_line)
#                         p.setFont("Helvetica", 11)
                    
#                     p.drawString(text_x, text_y, line)
#                     text_y -= 16
#             else:
#                 # Add space for paragraph breaks
#                 text_y -= 8

#         p.save()
#         buffer.seek(0)
        
#         print("✅ PDF generated successfully")
#         return StreamingResponse(
#             io.BytesIO(buffer.read()), 
#             media_type="application/pdf", 
#             headers={
#                 "Content-Disposition": "attachment; filename=audit_report.pdf"
#             }
#         )
        
#     except Exception as e:
#         print(f"❌ Error generating PDF: {str(e)}")
#         raise Exception(f"Failed to generate PDF: {str(e)}")

from reportlab.lib.pagesizes import landscape
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.utils import simpleSplit
import io
import os
from fastapi.responses import StreamingResponse

# === Custom Page Size and Margins ===
PAGE_WIDTH = 1000
PAGE_HEIGHT = 600
LEFT_MARGIN = inch
RIGHT_MARGIN = inch
TOP_MARGIN = 1.2 * inch
BOTTOM_MARGIN = inch

LOGO_WIDTH = 240
LOGO_HEIGHT = 45
LOGO_Y_OFFSET = PAGE_HEIGHT - TOP_MARGIN + 10

LOGO_PATH =os.path.join(BASE_DIR, "..", "assets", "Data_Vinci_Logo.png") # Make sure to update this to your actual path


def draw_header(c):
    logo_y = LOGO_Y_OFFSET

    if os.path.exists(LOGO_PATH):
        c.drawImage(LOGO_PATH, LEFT_MARGIN, LOGO_Y_OFFSET, width=LOGO_WIDTH, height=LOGO_HEIGHT, mask='auto')

    # Pink horizontal line after logo
    line_start = LEFT_MARGIN + LOGO_WIDTH + 10
    line_y = logo_y + LOGO_HEIGHT / 2
    c.setStrokeColor(colors.HexColor("#ef1fb3"))
    c.setLineWidth(4)
    c.line(line_start, line_y, PAGE_WIDTH - RIGHT_MARGIN, line_y)


def draw_footer_cta(c):
    sticker_text = "CLAIM YOUR FREE\nSTRATEGY SESSION"
    link_url = "https://datavinci.services/certified-google-analytics-consultants/?utm_source=ga4_audit&utm_medium=looker_report"

    sticker_x = PAGE_WIDTH - 250
    sticker_y = 25
    sticker_width = 180
    sticker_height = 40

    c.setFillColor(colors.HexColor("#007FFF"))
    c.roundRect(sticker_x, sticker_y, sticker_width, sticker_height, 8, stroke=0, fill=1)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(sticker_x + sticker_width / 2, sticker_y + 24, "CLAIM YOUR FREE")
    c.drawCentredString(sticker_x + sticker_width / 2, sticker_y + 12, "STRATEGY SESSION")

    c.linkURL(link_url, (sticker_x, sticker_y, sticker_x + sticker_width, sticker_y + sticker_height), relative=0)


def generate_pdf_report(section_title: str, content: str) -> StreamingResponse:
    try:
        print(f"📄 Generating PDF with title: {section_title}")

        if not content or not content.strip():
            content = "No content available for this report."

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))

        draw_header(c)

        # === LEFT SECTION (Heading) ===
        left_section_width = PAGE_WIDTH * 0.4
        c.setFillColor(colors.lightgrey)
        c.rect(LEFT_MARGIN, BOTTOM_MARGIN, left_section_width - LEFT_MARGIN, PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN, fill=True, stroke=False)

        # Draw title in left section
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 18)
        title_x = LEFT_MARGIN + 10
        title_y = PAGE_HEIGHT - TOP_MARGIN - 30
        title_lines = simpleSplit(section_title, "Helvetica-Bold", 18, left_section_width - 2 * inch)
        for i, line in enumerate(title_lines):
            c.drawString(title_x, title_y - i * 22, line)

        # === RIGHT SECTION (Content) ===
        text_x = left_section_width + 20
        text_y = PAGE_HEIGHT - TOP_MARGIN - 30
        text_width = PAGE_WIDTH - text_x - RIGHT_MARGIN
        c.setFont("Helvetica", 12)

        content_lines = content.strip().split('\n')
        for paragraph in content_lines:
            if paragraph.strip():
                wrapped_lines = simpleSplit(paragraph.strip(), "Helvetica", 12, text_width)
                for line in wrapped_lines:
                    if text_y < BOTTOM_MARGIN + 30:
                        c.showPage()
                        draw_header(c)
                        text_y = PAGE_HEIGHT - TOP_MARGIN - 30
                    c.drawString(text_x, text_y, line)
                    text_y -= 16
            else:
                text_y -= 8

        draw_footer_cta(c)

        c.save()
        buffer.seek(0)

        return StreamingResponse(
            io.BytesIO(buffer.read()),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=audit_report.pdf"}
        )

    except Exception as e:
        print(f"❌ Error generating PDF: {str(e)}")
        raise Exception(f"Failed to generate PDF: {str(e)}")
