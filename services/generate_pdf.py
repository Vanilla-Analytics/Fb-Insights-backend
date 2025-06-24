from reportlab.lib.pagesizes import landscape
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.utils import simpleSplit
import io
import os
from fastapi.responses import StreamingResponse
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PAGE_WIDTH = 1000
PAGE_HEIGHT = 600
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
def draw_metrics_grid(c, metrics, start_y):
    card_width = 180
    card_height = 60
    padding_x = 20
    padding_y = 20
    cols = 4
    x_start = LEFT_MARGIN
    y = start_y
    c.setFont("Helvetica-Bold", 14)

    for i, (label, value) in enumerate(metrics.items()):
                if "<" in label or "<" in value:
                    continue  # Skip HTML remnants

                # Clean "■" or bullets from value
                value_cleaned = str(value).replace("■", "").strip()

                col = i % cols
                row = i // cols
                x = x_start + col * (card_width + padding_x)
                y_offset = row * (card_height + padding_y)
                card_y = y - y_offset

                # Draw card
                c.setFillColor(colors.HexColor("#e1fbd2"))  # soft green background
                c.roundRect(x, card_y - card_height, card_width, card_height, 10, fill=1, stroke=0)

                # Centered metric title and value
                c.setFillColor(colors.HexColor("#222222"))
                c.setFont("Helvetica-Bold", 12)
                c.drawCentredString(x + card_width / 2, card_y - 18, label.strip())

                c.setFont("Helvetica", 12)
                c.drawCentredString(x + card_width / 2, card_y - 38, value_cleaned)

        

def generate_pdf_report(sections: list, ad_insights_df=None) -> StreamingResponse:
    try:
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))

        for i, section in enumerate(sections):
            draw_footer = True  # ✅ Set default at start of each section
            section_title = section.get("title", "Untitled Section")
            content = section.get("content", "No content available.")
            charts = section.get("charts", [])
            draw_header(c)

            if section_title.strip().upper() == "KEY METRICS":
                    # Page 1: Key Metrics Header & Cards
                    c.setFont("Helvetica-Bold", 24)
                    c.setFillColor(colors.black)
                    c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - TOP_MARGIN - 60, "Key Metrics")


                    metric_lines = [line for line in content.split("\n") if ":" in line and "Last 30" not in line]
                    metrics = dict(line.split(":", 1) for line in metric_lines)
                    draw_metrics_grid(c, metrics, PAGE_HEIGHT - 180) 

                    # Page 2: Trend Heading & Paragraph
                    c.showPage()
                    draw_header(c)

                    c.setFont("Helvetica-Bold", 20)
                    c.drawString(LEFT_MARGIN, PAGE_HEIGHT - TOP_MARGIN - 30, "Last 30 Days Trend Section")

                    paragraph = (
                        "The following section presents daily trend of the Key Metrics Identified in the previous section. "
                        "This helps the business analyse the daily variance in the business KPIs and also helps in correlating "
                        "how one metric affects the others."
                    )
                    c.setFont("Helvetica", 12)
                    #text_width = PAGE_WIDTH / 2
                    text_width = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
                    lines = simpleSplit(paragraph, "Helvetica", 12, text_width)
                    text_y = PAGE_HEIGHT - TOP_MARGIN - 60

                    for line in lines:
                        #c.drawRightString(PAGE_WIDTH - RIGHT_MARGIN, text_y, line)
                        c.drawString(LEFT_MARGIN, text_y, line)
                        text_y -= 14

                    # Page 3: Chart 1 — Amount Spent vs Purchase Conversion Value
                    if charts:
                        #c.showPage()
                        #draw_header(c)
                        try:
                            #chart_title = "Amount Spent vs Purchase Conversion Value"
                            chart_title = charts[0][0]
                            c.setFont("Helvetica-Bold", 16)
                            title_y = text_y - 10
                            #c.drawCentredString(PAGE_WIDTH / 2, title_y, chart_title)

                            chart_width = PAGE_WIDTH - 2 * LEFT_MARGIN
                            chart_height = 280
                            chart_x = (PAGE_WIDTH - chart_width) / 2
                            #chart_y = title_y - chart_height - 30 
                            #chart_y = max(BOTTOM_MARGIN + 40, title_y - chart_height - 30)
                            chart_y = max(BOTTOM_MARGIN + 60, title_y - chart_height - 10)



                            img1 = ImageReader(charts[0][1])
                            c.drawImage(img1, chart_x, chart_y, width=chart_width, height=chart_height, preserveAspectRatio=True)


                        except Exception as e:
                            print(f"⚠️ Chart 1 render error: {str(e)}")  

                    # Page 4: Chart 2 — Purchases vs ROAS
                    if len(charts) > 1:
                        c.showPage()
                        draw_header(c)
                        try:
                            chart_title = "Purchases vs ROAS"
                            c.setFont("Helvetica-Bold", 16)
                            title_y = text_y - 10
                            c.drawCentredString(PAGE_WIDTH / 2, title_y, chart_title)

                            chart_width = PAGE_WIDTH - 2 * LEFT_MARGIN
                            chart_height = 300
                            chart_x = (PAGE_WIDTH - chart_width) / 2
                            chart_y = title_y - chart_height - 30 

                            #img2 = ImageReader(charts[0][1])
                            img2 =   ImageReader(charts[1][1])
                            c.drawImage(img2, chart_x, chart_y, width=chart_width, height=chart_height, preserveAspectRatio=True)
                            # c.setFont("Helvetica-Bold", 14)
                            # #c.drawString(LEFT_MARGIN, PAGE_HEIGHT - TOP_MARGIN - 30, "Purchases vs ROAS")
                            # img2 = ImageReader(charts[1][1])
                            # #c.drawImage(img2, LEFT_MARGIN + 20, BOTTOM_MARGIN + 40, width=1000, height=300, preserveAspectRatio=True)
                            # # Page 4: Chart 2 — Heading + Centered Chart
                            # chart_title = "Purchases vs ROAS"
                            # c.setFont("Helvetica-Bold", 16)
                            # c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - TOP_MARGIN - 80, chart_title)

                            # chart_width = 1000
                            # chart_height = 300
                            # chart_x = (PAGE_WIDTH - chart_width) / 2
                            # chart_y = BOTTOM_MARGIN + 60

                            # c.drawImage(img2, chart_x, chart_y, width=chart_width, height=chart_height, preserveAspectRatio=True)

                        except Exception as e:
                            print(f"⚠️ Chart 2 render error: {str(e)}")  

                    if len(charts) > 2:
                        c.showPage()
                        draw_header(c)
                        try:
                            chart_title = "CPA vs Link CPC"
                            c.setFont("Helvetica-Bold", 16)
                            title_y = PAGE_HEIGHT - TOP_MARGIN - 80
                            c.drawCentredString(PAGE_WIDTH / 2, title_y, chart_title)

                            chart_width = PAGE_WIDTH - 2 * LEFT_MARGIN
                            chart_height = 300
                            chart_x = (PAGE_WIDTH - chart_width) / 2
                            chart_y = BOTTOM_MARGIN + 40

                            img3 = ImageReader(charts[2][1])
                            c.drawImage(img3, chart_x, chart_y, width=chart_width, height=chart_height, preserveAspectRatio=True)
                        except Exception as e:
                            print(f"⚠️ Chart 3 render error: {str(e)}")

                    if len(charts) > 3:
                        c.showPage()
                        draw_header(c)
                        try:
                            chart_title = "Click to Conversion vs CTR"
                            c.setFont("Helvetica-Bold", 16)
                            title_y = PAGE_HEIGHT - TOP_MARGIN - 80
                            c.drawCentredString(PAGE_WIDTH / 2, title_y, chart_title)

                            chart_width = PAGE_WIDTH - 2 * LEFT_MARGIN
                            chart_height = 300
                            chart_x = (PAGE_WIDTH - chart_width) / 2
                            chart_y = BOTTOM_MARGIN + 40

                            img4 = ImageReader(charts[3][1])
                            c.drawImage(img4, chart_x, chart_y, width=chart_width, height=chart_height, preserveAspectRatio=True)
                        except Exception as e:
                            print(f"⚠️ Chart 4 render error: {str(e)}")

                    
                    # New Page: Full Table Summary
                    if ad_insights_df is not None and not ad_insights_df.empty:
                        c.showPage()
                        draw_header(c)
                        
                        # title_y = PAGE_HEIGHT - TOP_MARGIN - 100
                        # c.setFont("Helvetica-Bold", 16)
                        # c.drawCentredString(PAGE_WIDTH / 2, title_y, "Campaign Performance Summary")

                        # Prepare table data
                        table_data = [["Day", "Amount spent", "Purchases", "Purchases conversion value", "CPA", "Impressions","CTR", "Link clicks", "Click To Conversion", "ROAS"]]

                        import pandas as pd

                        for _, row in ad_insights_df.iterrows():
                            table_data.append([
                                pd.to_datetime(row['date']).strftime("%d %b %Y"),
                                f"${row['spend']:,.2f}",
                                int(row['purchases']),
                                f"${row['purchase_value']:,.2f}",
                                f"${row['cpa']:,.2f}",
                                f"{int(row['impressions']):,}",
                                f"{row['ctr']:.2%}",
                                int(row['clicks']),
                                f"{row['click_to_conversion']:.2%}",
                                f"{row['roas']:.2f}",
                            ])
                        print("🖨 PDF row date:", row['date'], type(row['date']))

                        # Calculate grand totals
                        totals = {
                            'spend': ad_insights_df['spend'].sum(),
                            'purchases': ad_insights_df['purchases'].sum(),
                            'purchase_value': ad_insights_df['purchase_value'].sum(),
                            'cpa': ad_insights_df['cpa'].mean(),  # or weighted average
                            'impressions': ad_insights_df['impressions'].sum(),
                            'ctr': ad_insights_df['ctr'].mean(),
                            'clicks': ad_insights_df['clicks'].sum(),
                            'click_to_conversion': ad_insights_df['click_to_conversion'].mean(),
                            'roas': ad_insights_df['roas'].mean()
                        }

                        # Append grand total row
                        table_data.append([
                            "Grand Total",
                            f"${totals['spend']:,.2f}",
                            int(totals['purchases']),
                            f"${totals['purchase_value']:,.2f}",
                            f"${totals['cpa']:,.2f}",
                            f"{int(totals['impressions']):,}",
                            f"{totals['ctr']:.2%}",
                            int(totals['clicks']),
                            f"{totals['click_to_conversion']:.2%}",
                            f"{totals['roas']:.2f}",
                        ])

                        # Limit row count if needed (for fitting one page), or use page breaks
                        summary_table = Table(table_data, repeatRows=1, colWidths=[90]*10)
                        summary_table.setStyle(TableStyle([

                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, -1), 8),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("BACKGROUND", (0, -1), (-1, -1), colors.lightblue),  # Last row = Grand Total
                            ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold")
                        ]))

                        # title_y = PAGE_HEIGHT - TOP_MARGIN - 100
                        # table_max_height = PAGE_HEIGHT - TOP_MARGIN - 100
                        # estimated_height = 15 * len(table_data[:30])
                        # table_y = table_max_height - estimated_height
                        # table_y = max(BOTTOM_MARGIN + 60, PAGE_HEIGHT / 2 - estimated_height / 2)

                        # Reserve fixed margin from top
                        max_table_height = PAGE_HEIGHT - TOP_MARGIN - 80  # Header space
                        min_table_y = BOTTOM_MARGIN + 50
                        estimated_height = 16 * len(table_data[:30])
                        #table_y = max(min_table_y, max_table_height - estimated_height)
                        table_y = max(BOTTOM_MARGIN + 30, PAGE_HEIGHT - TOP_MARGIN - estimated_height - 20)




                        summary_table.wrapOn(c, PAGE_WIDTH, PAGE_HEIGHT)
                        summary_table.drawOn(c, LEFT_MARGIN, table_y)

                        #draw_footer = False  # Skip footer for table page

                        # ✅ New Page: Campaign Level Performance Table
                    if ad_insights_df is not None and 'campaign_name' in ad_insights_df.columns:
                        c.showPage()
                        draw_header(c)

                        c.setFont("Helvetica-Bold", 16)
                        c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - TOP_MARGIN - 30, "Campaign Level Performance")

                        grouped_campaigns = ad_insights_df.groupby('campaign_name').agg({
                            'spend': 'sum',
                            'purchase_value': 'sum',
                            'purchases': 'sum'
                        }).reset_index()

                        grouped_campaigns['roas'] = grouped_campaigns['purchase_value'] / grouped_campaigns['spend'].replace(0, 1)
                        grouped_campaigns['cpa'] = grouped_campaigns['spend'] / grouped_campaigns['purchases'].replace(0, 1)

                        table_data = [["Campaign Name", "Amount Spent", "Revenue", "Purchases", "ROAS", "CPA"]]

                        for _, row in grouped_campaigns.iterrows():
                            table_data.append([
                                row['campaign_name'],
                                f"${row['spend']:,.2f}",
                                f"${row['purchase_value']:,.2f}",
                                int(row['purchases']),
                                f"{row['roas']:.2f}",
                                f"${row['cpa']:.2f}"
                            ])

                        # Grand total row
                        grand_totals = {
                            'spend': grouped_campaigns['spend'].sum(),
                            'purchase_value': grouped_campaigns['purchase_value'].sum(),
                            'purchases': grouped_campaigns['purchases'].sum(),
                            'roas': grouped_campaigns['purchase_value'].sum() / grouped_campaigns['spend'].replace(0, 1).sum(),
                            'cpa': grouped_campaigns['spend'].sum() / grouped_campaigns['purchases'].replace(0, 1).sum()
                        }

                        table_data.append([
                            "Grand Total",
                            f"${grand_totals['spend']:,.2f}",
                            f"${grand_totals['purchase_value']:,.2f}",
                            int(grand_totals['purchases']),
                            f"{grand_totals['roas']:.2f}",
                            f"${grand_totals['cpa']:.2f}"
                        ])

                        performance_table = Table(table_data, repeatRows=1, colWidths=[170, 90, 90, 80, 80, 80])
                        performance_table.setStyle(TableStyle([
                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, -1), 8),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("BACKGROUND", (0, -1), (-1, -1), colors.lightblue),
                            ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold")
                        ]))

                        table_y = PAGE_HEIGHT - TOP_MARGIN - 100
                        performance_table.wrapOn(c, PAGE_WIDTH, PAGE_HEIGHT)
                        performance_table.drawOn(c, LEFT_MARGIN, table_y)



                    else:
                        c.showPage()
                        draw_header(c)
                        c.setFont("Helvetica-Bold", 16)
                        c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - TOP_MARGIN - 30, "Campaign Performance Summary")
                        c.setFont("Helvetica", 12)
                        c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT / 2, "⚠ No ad data available to display the summary table.") 
                    
             
                    
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

            #draw_footer_cta(c)
            if draw_footer:
                draw_footer_cta(c)

            if i < len(sections) - 1:
                c.showPage()

        c.save()
        buffer.seek(0)
        return StreamingResponse(io.BytesIO(buffer.read()), media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=audit_report.pdf"})

    except Exception as e:
        print(f"❌ Error generating PDF: {str(e)}")
        raise Exception(f"Failed to generate PDF: {str(e)}")
