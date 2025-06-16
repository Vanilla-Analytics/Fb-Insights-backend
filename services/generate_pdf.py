# services/generate_pdf.py
# services/generate_pdf.py
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.utils import simpleSplit
import io
from fastapi.responses import StreamingResponse


def generate_pdf_report(section_title: str, content: str) -> StreamingResponse:
    """Generate a PDF report with the given title and content"""
    try:
        print(f"📄 Generating PDF with title: {section_title}")
        
        if not content or not content.strip():
            content = "No content available for this report."
        
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
        
        # Handle long titles by wrapping them
        title_lines = simpleSplit(section_title, "Helvetica-Bold", 18, left_width - 30)
        for i, title_line in enumerate(title_lines):
            p.drawString(title_x, title_y - (i * 22), title_line)

        # === RIGHT SECTION (70%) ===
        text_x = margin + left_width + 20
        text_y = height - margin - 40
        text_width = width - text_x - margin

        # Wrap and draw content
        p.setFont("Helvetica", 11)
        content_lines = content.strip().split('\n')
        
        for paragraph in content_lines:
            if paragraph.strip():  # Skip empty lines
                wrapped_lines = simpleSplit(paragraph.strip(), "Helvetica", 11, text_width)
                for line in wrapped_lines:
                    if text_y < margin + 30:
                        p.showPage()
                        text_y = height - margin - 40
                        # Re-draw left section on each new page
                        p.setFillColor(colors.lightgrey)
                        p.rect(margin, margin, left_width - margin, height - 2 * margin, fill=True, stroke=False)
                        p.setFillColor(colors.black)
                        p.setFont("Helvetica-Bold", 18)
                        for i, title_line in enumerate(title_lines):
                            p.drawString(title_x, height - margin - 40 - (i * 22), title_line)
                        p.setFont("Helvetica", 11)
                    
                    p.drawString(text_x, text_y, line)
                    text_y -= 16
            else:
                # Add space for paragraph breaks
                text_y -= 8

        p.save()
        buffer.seek(0)
        
        print("✅ PDF generated successfully")
        return StreamingResponse(
            io.BytesIO(buffer.read()), 
            media_type="application/pdf", 
            headers={
                "Content-Disposition": "attachment; filename=audit_report.pdf"
            }
        )
        
    except Exception as e:
        print(f"❌ Error generating PDF: {str(e)}")
        raise Exception(f"Failed to generate PDF: {str(e)}")