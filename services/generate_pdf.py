# from reportlab.lib.pagesizes import landscape
# from reportlab.pdfgen import canvas
# from reportlab.lib import colors
# from reportlab.lib.units import inch
# from reportlab.lib.utils import simpleSplit
# import io
# import os
# from fastapi.responses import StreamingResponse
# from reportlab.lib.utils import ImageReader
# import re

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# PAGE_WIDTH = 1000
# PAGE_HEIGHT = 500
# LEFT_MARGIN = inch
# RIGHT_MARGIN = inch
# TOP_MARGIN = 1.2 * inch
# BOTTOM_MARGIN = inch

# LOGO_WIDTH = 240
# LOGO_HEIGHT = 45
# LOGO_Y_OFFSET = PAGE_HEIGHT - TOP_MARGIN + 10

# LOGO_PATH = os.path.join(BASE_DIR, "..", "assets", "Data_Vinci_Logo.png")

# def parse_bold_segments(text):
#     segments = []
#     parts = re.split(r'(\*\*.*?\*\*)', text)
#     for part in parts:
#         if part.startswith('**') and part.endswith('**'):
#             segments.append((part[2:-2], True))
#         else:
#             segments.append((part, False))
#     return segments

# def draw_header(c):
#     logo_y = LOGO_Y_OFFSET
#     if os.path.exists(LOGO_PATH):
#         c.drawImage(LOGO_PATH, LEFT_MARGIN, LOGO_Y_OFFSET, width=LOGO_WIDTH, height=LOGO_HEIGHT, mask='auto')
#     line_start = LEFT_MARGIN + LOGO_WIDTH + 10
#     line_y = logo_y + LOGO_HEIGHT / 2
#     c.setStrokeColor(colors.HexColor("#ef1fb3"))
#     c.setLineWidth(4)
#     c.line(line_start, line_y, PAGE_WIDTH - RIGHT_MARGIN, line_y)

# def draw_footer_cta(c):
#     link_url = "https://datavinci.services/certified-google-analytics-consultants/?utm_source=ga4_audit&utm_medium=looker_report"
#     sticker_x = PAGE_WIDTH - 250
#     sticker_y = 25
#     sticker_width = 180
#     sticker_height = 40
#     c.setFillColor(colors.HexColor("#007FFF"))
#     c.roundRect(sticker_x, sticker_y, sticker_width, sticker_height, 8, stroke=0, fill=1)
#     c.setFillColor(colors.white)
#     c.setFont("Helvetica-Bold", 12)
#     c.drawCentredString(sticker_x + sticker_width / 2, sticker_y + 24, "CLAIM YOUR FREE")
#     c.drawCentredString(sticker_x + sticker_width / 2, sticker_y + 12, "STRATEGY SESSION")
#     c.linkURL(link_url, (sticker_x, sticker_y, sticker_x + sticker_width, sticker_y + sticker_height), relative=0)

# def draw_metrics_grid(c, metrics, start_y):
#     card_width = 200
#     card_height = 50
#     padding_x = 20
#     padding_y = 20
#     cols = 3
#     x_start = LEFT_MARGIN
#     y = start_y
#     c.setFont("Helvetica-Bold", 14)

#     for i, (label, value) in enumerate(metrics.items()):
#         col = i % cols
#         row = i // cols
#         x = x_start + col * (card_width + padding_x)
#         y_offset = row * (card_height + padding_y)
#         card_y = y - y_offset

#         c.setFillColor(colors.HexColor("#d4fcd7"))
#         c.roundRect(x, card_y - card_height, card_width, card_height, 6, fill=1, stroke=0)
#         c.setFillColor(colors.black)
#         c.drawString(x + 10, card_y - 18, label)
#         c.setFont("Helvetica", 12)
#         c.drawString(x + 10, card_y - 36, str(value))
#         c.setFont("Helvetica-Bold", 14)

# def generate_pdf_report(sections: list) -> StreamingResponse:
#     try:
#         buffer = io.BytesIO()
#         c = canvas.Canvas(buffer, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))

#         for i, section in enumerate(sections):
#             section_title = section.get("title", "Untitled Section")
#             content = section.get("content", "No content available.")
#             charts = section.get("charts", [])
#             draw_header(c)

#             if section_title.strip().upper() == "KEY METRICS":
#                 c.setFont("Helvetica-Bold", 24)
#                 c.setFillColor(colors.black)
#                 c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - TOP_MARGIN - 10, "Key Metrics")

#                 # Parse metrics from content string
#                 metric_lines = [line for line in content.split("\n") if ":" in line and "Last 30" not in line]
#                 metrics = dict(line.split(":", 1) for line in metric_lines)

#                 draw_metrics_grid(c, metrics, PAGE_HEIGHT - TOP_MARGIN - 50)

#                 c.setFont("Helvetica-Bold", 16)
#                 c.drawString(LEFT_MARGIN, PAGE_HEIGHT - TOP_MARGIN - 220, "Last 30 Days Trend Section")
#                 c.setFont("Helvetica", 12)
#                 c.drawString(LEFT_MARGIN, PAGE_HEIGHT - TOP_MARGIN - 240, "The following section presents daily trend of the Key Metrics to help analyze business KPIs.")

#                 chart_y = BOTTOM_MARGIN + 60
#                 for chart_title, chart_buf in charts:
#                     try:
#                         img = ImageReader(chart_buf)
#                         c.drawImage(img, LEFT_MARGIN + 80, chart_y, width=800, height=180, preserveAspectRatio=True)
#                         chart_y += 200
#                     except Exception as e:
#                         print(f"⚠️ Could not render chart '{chart_title}': {str(e)}")
#             else:
#                 # Default layout
#                 left_section_width = PAGE_WIDTH * 0.4
#                 c.setFillColor(colors.white)
#                 c.rect(LEFT_MARGIN, BOTTOM_MARGIN, left_section_width - LEFT_MARGIN, PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN, fill=True, stroke=False)
#                 title_width = left_section_width - 2 * inch
#                 title_lines = simpleSplit(section_title, "Helvetica-Bold", 20, title_width)
#                 title_y = PAGE_HEIGHT - TOP_MARGIN - 20
#                 c.setFillColor(colors.black)
#                 c.setFont("Helvetica-Bold", 22)
#                 title_line_height = 24
#                 title_block_height = title_line_height * len(title_lines)
#                 title_y_start = BOTTOM_MARGIN + (PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN + title_block_height) / 2 - title_line_height
#                 for line in title_lines:
#                     c.drawString(LEFT_MARGIN + 10, title_y_start, line)
#                     title_y_start -= title_line_height

#                 text_x = left_section_width + 20
#                 text_width = PAGE_WIDTH - text_x - RIGHT_MARGIN
#                 c.setStrokeColor(colors.HexColor("#007bff"))
#                 c.setLineWidth(8)
#                 c.line(text_x - 10, BOTTOM_MARGIN, text_x - 10, PAGE_HEIGHT - TOP_MARGIN)

#                 #text_y = PAGE_HEIGHT - TOP_MARGIN - 30
#                 text_y = PAGE_HEIGHT - 110  # Reduces top margin

#                 c.setFont("Helvetica", 14)
#                 content_lines = content.strip().split('\n')
#                 for paragraph in content_lines:
#                     if paragraph.strip():
#                         wrapped_lines = simpleSplit(paragraph.strip(), "Helvetica", 14, text_width)
#                         for line in wrapped_lines:
#                             #if text_y < BOTTOM_MARGIN + 30:
#                             if text_y < 40:
#                                 c.showPage()
#                                 draw_header(c)
#                                 text_y = PAGE_HEIGHT - TOP_MARGIN - 30
#                             x_cursor = text_x
#                             for seg_text, is_bold in parse_bold_segments(line):
#                                 font_name = "Helvetica-Bold" if is_bold else "Helvetica"
#                                 c.setFont(font_name, 14)
#                                 c.drawString(x_cursor, text_y, seg_text)
#                                 x_cursor += c.stringWidth(seg_text, font_name, 14)
#                             text_y -= 20
#                     else:
#                         text_y -= 8

#                 chart_y = BOTTOM_MARGIN + 40
#                 for chart_title, chart_buf in charts:
#                     try:
#                         img = ImageReader(chart_buf)
#                         c.drawImage(img, LEFT_MARGIN + 300, chart_y, width=500, height=240, preserveAspectRatio=True)
#                         chart_y += 260
#                     except Exception as e:
#                         print(f"⚠️ Could not render chart '{chart_title}': {str(e)}")

#             draw_footer_cta(c)
#             if i < len(sections) - 1:
#                 c.showPage()

#         c.save()
#         buffer.seek(0)
#         return StreamingResponse(io.BytesIO(buffer.read()), media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=audit_report.pdf"})

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
from reportlab.lib.utils import ImageReader
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

def parse_bold_segments(text):
    segments = []
    parts = re.split(r'(\*\*.*?\*\*)', text)
    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            segments.append((part[2:-2], True))
        else:
            segments.append((part, False))
    return segments

def draw_header(c):
    logo_y = LOGO_Y_OFFSET
    if os.path.exists(LOGO_PATH):
        c.drawImage(LOGO_PATH, LEFT_MARGIN, LOGO_Y_OFFSET, width=LOGO_WIDTH, height=LOGO_HEIGHT, mask='auto')
    line_start = LEFT_MARGIN + LOGO_WIDTH + 10
    line_y = logo_y + LOGO_HEIGHT / 2
    c.setStrokeColor(colors.HexColor("#ef1fb3"))
    c.setLineWidth(4)
    c.line(line_start, line_y, PAGE_WIDTH - RIGHT_MARGIN, line_y)

def draw_footer_cta(c):
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

def draw_metrics_grid(c, metrics, start_y):
    card_width = 200
    card_height = 50
    padding_x = 20
    padding_y = 20
    cols = 3
    x_start = LEFT_MARGIN
    y = start_y
    c.setFont("Helvetica-Bold", 14)

    for i, (label, value) in enumerate(metrics.items()):
        col = i % cols
        row = i // cols
        x = x_start + col * (card_width + padding_x)
        y_offset = row * (card_height + padding_y)
        card_y = y - y_offset

        c.setFillColor(colors.HexColor("#d4fcd7"))
        c.roundRect(x, card_y - card_height, card_width, card_height, 6, fill=1, stroke=0)
        c.setFillColor(colors.black)
        c.drawString(x + 10, card_y - 18, label)
        c.setFont("Helvetica", 12)
        c.drawString(x + 10, card_y - 36, str(value))
        c.setFont("Helvetica-Bold", 14)

def generate_pdf_report(sections: list) -> StreamingResponse:
    try:
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))

        for i, section in enumerate(sections):
            section_title = section.get("title", "Untitled Section")
            content = section.get("content", "No content available.")
            charts = section.get("charts", [])
            draw_header(c)

            
            if section_title.strip().upper() == "KEY METRICS":
                c.setFont("Helvetica-Bold", 24)
                c.setFillColor(colors.black)
                c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - TOP_MARGIN - 10, "Key Metrics")

                c.setFont("Helvetica", 12)
                c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - TOP_MARGIN - 30, "Performance overview of your campaign across key indicators.")

                # Parse metrics from content string
                metric_lines = [line for line in content.split("
") if ":" in line and "Last 30" not in line]
                metrics = dict(line.split(":", 1) for line in metric_lines)

                def draw_metrics_grid_ui(c, metrics, start_y):
                    card_width = 220
                    card_height = 60
                    padding_x = 30
                    padding_y = 25
                    cols = 3
                    x_start = LEFT_MARGIN
                    y = start_y
                    c.setFont("Helvetica-Bold", 14)

                    for i, (label, value) in enumerate(metrics.items()):
                        col = i % cols
                        row = i // cols
                        x = x_start + col * (card_width + padding_x)
                        y_offset = row * (card_height + padding_y)
                        card_y = y - y_offset

                        c.setFillColor(colors.HexColor("#e1fbd2"))
                        c.roundRect(x, card_y - card_height, card_width, card_height, 10, fill=1, stroke=0)
                        c.setFillColor(colors.black)
                        c.setFont("Helvetica-Bold", 12)
                        c.drawCentredString(x + card_width / 2, card_y - 20, label)
                        c.setFont("Helvetica", 12)
                        c.drawCentredString(x + card_width / 2, card_y - 38, str(value))

                draw_metrics_grid_ui(c, metrics, PAGE_HEIGHT - TOP_MARGIN - 60)

                c.setFont("Helvetica-Bold", 18)
                c.setFillColor(colors.black)
                c.drawString(LEFT_MARGIN, PAGE_HEIGHT - TOP_MARGIN - 230, "Last 30 Days Trend Section")
                c.setFont("Helvetica", 12)
                c.drawString(LEFT_MARGIN, PAGE_HEIGHT - TOP_MARGIN - 250, "The following section presents daily trend of the Key Metrics to help analyze business KPIs.")

                chart_y = BOTTOM_MARGIN + 100
                for chart_title, chart_buf in charts:
                    try:
                        img = ImageReader(chart_buf)
                        c.drawImage(img, LEFT_MARGIN + 50, chart_y, width=860, height=180, preserveAspectRatio=True)
                        chart_y += 200
                    except Exception as e:
                        print(f"⚠️ Could not render chart '{chart_title}': {str(e)}")

            else:
                # Default layout
                left_section_width = PAGE_WIDTH * 0.4
                c.setFillColor(colors.white)
                c.rect(LEFT_MARGIN, BOTTOM_MARGIN, left_section_width - LEFT_MARGIN, PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN, fill=True, stroke=False)
                title_width = left_section_width - 2 * inch
                title_lines = simpleSplit(section_title, "Helvetica-Bold", 20, title_width)
                title_y = PAGE_HEIGHT - TOP_MARGIN - 20
                c.setFillColor(colors.black)
                c.setFont("Helvetica-Bold", 22)
                title_line_height = 24
                title_block_height = title_line_height * len(title_lines)
                title_y_start = BOTTOM_MARGIN + (PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN + title_block_height) / 2 - title_line_height
                for line in title_lines:
                    c.drawString(LEFT_MARGIN + 10, title_y_start, line)
                    title_y_start -= title_line_height

                text_x = left_section_width + 20
                text_width = PAGE_WIDTH - text_x - RIGHT_MARGIN
                c.setStrokeColor(colors.HexColor("#007bff"))
                c.setLineWidth(8)
                c.line(text_x - 10, BOTTOM_MARGIN, text_x - 10, PAGE_HEIGHT - TOP_MARGIN)

                #text_y = PAGE_HEIGHT - TOP_MARGIN - 30
                text_y = PAGE_HEIGHT - 110  # Reduces top margin

                c.setFont("Helvetica", 14)
                content_lines = content.strip().split('\n')
                for paragraph in content_lines:
                    if paragraph.strip():
                        wrapped_lines = simpleSplit(paragraph.strip(), "Helvetica", 14, text_width)
                        for line in wrapped_lines:
                            #if text_y < BOTTOM_MARGIN + 30:
                            if text_y < 40:
                                c.showPage()
                                draw_header(c)
                                text_y = PAGE_HEIGHT - TOP_MARGIN - 30
                            x_cursor = text_x
                            for seg_text, is_bold in parse_bold_segments(line):
                                font_name = "Helvetica-Bold" if is_bold else "Helvetica"
                                c.setFont(font_name, 14)
                                c.drawString(x_cursor, text_y, seg_text)
                                x_cursor += c.stringWidth(seg_text, font_name, 14)
                            text_y -= 20
                    else:
                        text_y -= 8

                chart_y = BOTTOM_MARGIN + 40
                for chart_title, chart_buf in charts:
                    try:
                        img = ImageReader(chart_buf)
                        c.drawImage(img, LEFT_MARGIN + 300, chart_y, width=500, height=240, preserveAspectRatio=True)
                        chart_y += 260
                    except Exception as e:
                        print(f"⚠️ Could not render chart '{chart_title}': {str(e)}")

            draw_footer_cta(c)
            if i < len(sections) - 1:
                c.showPage()

        c.save()
        buffer.seek(0)
        return StreamingResponse(io.BytesIO(buffer.read()), media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=audit_report.pdf"})

    except Exception as e:
        print(f"❌ Error generating PDF: {str(e)}")
        raise Exception(f"Failed to generate PDF: {str(e)}")
