from reportlab.lib.pagesizes import landscape
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.utils import simpleSplit
import io
import os
from fastapi.responses import StreamingResponse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === Custom Page Size and Margins ===
PAGE_WIDTH = 1000
PAGE_HEIGHT = 500
LEFT_MARGIN = inch
RIGHT_MARGIN = inch
TOP_MARGIN = 1.2 * inch
BOTTOM_MARGIN = inch

LOGO_WIDTH = 240
LOGO_HEIGHT = 45
LOGO_Y_OFFSET = PAGE_HEIGHT - TOP_MARGIN + 10

LOGO_PATH = os.path.join(BASE_DIR, "..", "assets", "Data_Vinci_Logo.png")

import re

def parse_bold_segments(text):
    """
    Convert markdown-style bold (**text**) to tuples of (text, is_bold)
    Example: "This is **bold** text" → [('This is ', False), ('bold', True), (' text', False)]
    """
    segments = []
    parts = re.split(r'(\*\*.*?\*\*)', text)
    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            segments.append((part[2:-2], True))  # remove **
        else:
            segments.append((part, False))
    return segments

def clean_text(text):
    """
    Removes e.g., eg., etc. and trims whitespace
    """
    return re.sub(r'\b(e\.g\.,?|eg\.?)\b', '', text, flags=re.IGNORECASE).strip()

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


def calculate_content_height(content_lines, font_name, font_size, text_width):
    """Calculate the total height needed for the content"""
    total_height = 0
    for paragraph in content_lines:
        if paragraph.strip():
            wrapped_lines = simpleSplit(paragraph.strip(), font_name, font_size, text_width)
            total_height += len(wrapped_lines) * 22  # 16 is line spacing
        else:
            total_height += 8  # paragraph break spacing
    return total_height


def calculate_title_height(section_title, font_name, font_size, title_width):
    """Calculate the total height needed for the title"""
    title_lines = simpleSplit(section_title, font_name, font_size, title_width)
    return len(title_lines) * 22  # 22 is title line spacing


def generate_pdf_report(sections: list) -> StreamingResponse:
    try:
        print(f"📄 Generating PDF with {len(sections)} sections")

        if not sections:
            sections = [{"title": "No Content", "content": "No content available for this report."}]

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))

        # Generate each section on its own page
        for i, section in enumerate(sections):
            section_title = section.get("title", "Untitled Section")
            content = section.get("content", "No content available.")
            
            if not content or not content.strip():
                content = "No content available for this section."

            # Draw header for each page
            draw_header(c)

            # === Calculate available space ===
            available_height = PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN - 60  # 60 for some padding

            # === LEFT SECTION (Heading) ===
            left_section_width = PAGE_WIDTH * 0.4
            c.setFillColor(colors.white)
            c.rect(LEFT_MARGIN, BOTTOM_MARGIN, left_section_width - LEFT_MARGIN, PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN, fill=True, stroke=False)

            # Calculate title dimensions
            title_width = left_section_width - 2 * inch
            title_height = calculate_title_height(section_title, "Helvetica-Bold", 20, title_width)
            
            # Center the title vertically
            title_start_y = BOTTOM_MARGIN + (available_height - title_height) / 2 + title_height
            
            # Draw title in left section
            c.setFillColor(colors.black)
            c.setFont("Helvetica-Bold", 20)
            title_x = LEFT_MARGIN + 10
            title_lines = simpleSplit(section_title, "Helvetica-Bold", 20, title_width)
            for j, line in enumerate(title_lines):
                c.drawString(title_x, title_start_y - j * 22, line)


            # === RIGHT SECTION (Content) ===
            text_x = left_section_width + 20
            text_width = PAGE_WIDTH - text_x - RIGHT_MARGIN

            # Draw blue vertical divider line
            c.setStrokeColor(colors.HexColor("#007bff"))  # Blue color
            c.setLineWidth(8)
            c.line(text_x - 10, BOTTOM_MARGIN, text_x - 10, PAGE_HEIGHT - TOP_MARGIN)
            
            # Calculate content dimensions
            content_lines = content.strip().split('\n')
            content_height = calculate_content_height(content_lines, "Helvetica", 12, text_width)
            
            # Center the content vertically
            text_start_y = BOTTOM_MARGIN + (available_height - content_height) / 2 + content_height
            text_y = text_start_y
            
            c.setFont("Helvetica", 16)

            for paragraph in content_lines:
                if paragraph.strip():
                    wrapped_lines = simpleSplit(paragraph.strip(), "Helvetica", 16, text_width)
                    for line in wrapped_lines:

                        if text_y < BOTTOM_MARGIN + 30:
                            c.showPage()
                            draw_header(c)
                            text_y = text_start_y

                        x_cursor = text_x
                        for seg_text, is_bold in parse_bold_segments(line):
                            font_name = "Helvetica-Bold" if is_bold else "Helvetica"
                            c.setFont(font_name, 16)
                            c.drawString(x_cursor, text_y, seg_text)
                            x_cursor += c.stringWidth(seg_text, font_name, 16)

                        text_y -= 20  # Adjust spacing for new font size

                else:
                    text_y -= 8

            # Draw footer CTA on each page
            draw_footer_cta(c)

            # Add new page if not the last section
            if i < len(sections) - 1:
                c.showPage()

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