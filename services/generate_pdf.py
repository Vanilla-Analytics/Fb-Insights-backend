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
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

font_path = os.path.join("assets", "fonts", "DejaVuSans.ttf")
pdfmetrics.registerFont(TTFont("DejaVuSans", font_path))

PAGE_WIDTH = 1000
PAGE_HEIGHT = 700
LEFT_MARGIN = inch
RIGHT_MARGIN = inch
TOP_MARGIN = 1.2 * inch
BOTTOM_MARGIN = inch

LOGO_WIDTH = 240
LOGO_HEIGHT = 45
LOGO_Y_OFFSET = PAGE_HEIGHT - TOP_MARGIN + 10

LOGO_PATH = os.path.join(BASE_DIR, "..", "assets", "Data_Vinci_Logo.png")
# def adjust_page_height(c, section: dict):
#     """
#     Adjust PAGE_HEIGHT based on whether the section contains a table (no charts).
#     - If charts are present → PAGE_HEIGHT = 600
#     - If no charts, but tables are drawn → PAGE_HEIGHT = 1000
#     """
#     global PAGE_HEIGHT, LOGO_Y_OFFSET, TOP_MARGIN

#     # More explicit condition for table pages
#     is_table_page = (
#         "Daily Campaign Performance Summary" in section.get("title", "") or 
#         "CAMPAIGN PERFORMANCE SUMMARY" in section.get("title", "") or
#         section.get("contains_table", False)
#     )
#     if is_table_page:
#         PAGE_HEIGHT = 1400
#     else:
#         PAGE_HEIGHT = 600

#     LOGO_Y_OFFSET = PAGE_HEIGHT - TOP_MARGIN + 10
#     c.setPageSize((PAGE_WIDTH, PAGE_HEIGHT))
def set_font_with_currency(c, currency_symbol, fallback_font="Helvetica", size=12):
    if currency_symbol == "₹":
        c.setFont("DejaVuSans", size)
    else:
        c.setFont(fallback_font, size)

def adjust_page_height(c, section: dict):
    """
    Adjust PAGE_HEIGHT based on section title/content.
    - "Campaign Performance Summary" → PAGE_HEIGHT = 1800
    - "Daily Campaign Performance Summary" → PAGE_HEIGHT = 1400
    - Else → PAGE_HEIGHT = 600
    """
    global PAGE_HEIGHT, LOGO_Y_OFFSET, TOP_MARGIN

    #title = section.get("title", "").upper()
    title = section.get("title", "").upper().strip()

    # if "Campaign Performance Overview" in title:
    #      PAGE_HEIGHT = 1400
    # elif title.strip().upper() == "CAMPAIGN PERFORMANCE SUMMARY" :
    #     PAGE_HEIGHT = 1800
    # elif title.strip().upper() == "3 CHARTS SECTION":
    #     PAGE_HEIGHT = 1400
    # else:
    #     PAGE_HEIGHT = 600
    if "CAMPAIGN PERFORMANCE OVERVIEW" in title:
        PAGE_HEIGHT = 1300
    elif title == "CAMPAIGN PERFORMANCE SUMMARY":
        PAGE_HEIGHT = 1900
    elif title == "3 CHARTS SECTION":
        PAGE_HEIGHT = 1400
    else:
        PAGE_HEIGHT = 600

    LOGO_Y_OFFSET = PAGE_HEIGHT - TOP_MARGIN + 10
    c.setPageSize((PAGE_WIDTH, PAGE_HEIGHT))


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
                
                set_font_with_currency(c, value_cleaned.strip()[0], size=12)
                c.drawCentredString(x + card_width / 2, card_y - 38, value_cleaned)


                # c.setFont("Helvetica", 12)
                # c.drawCentredString(x + card_width / 2, card_y - 38, value_cleaned)

        

def generate_pdf_report(sections: list, ad_insights_df=None,full_ad_insights_df=None, currency_symbol=None, split_charts=None) -> StreamingResponse:
    global PAGE_HEIGHT, LOGO_Y_OFFSET, TOP_MARGIN  # ✅ Fixes UnboundLocalError

    if currency_symbol is None:
        currency_symbol = "₹"
    try:
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer)   
        #c = canvas.Canvas(buffer, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))

        for i, section in enumerate(sections):  
            
            adjust_page_height(c, section)  # Adjust page height based on section title
            section_title = section.get("title", "Untitled Section")
            
            draw_footer = True  # ✅ Set default at start of each section
            section_title = section.get("title", "Untitled Section")
            content = section.get("content", "No content available.")
            charts = section.get("charts", [])
            draw_header(c)

            if section_title.strip().upper() == "KEY METRICS":
                    # Debug print to verify currency symbol
                    print(f"🔎 Current currency symbol: {currency_symbol}")
                    # Page 1: Key Metrics Header & Cards
                    c.setFont("Helvetica-Bold", 24)
                    c.setFillColor(colors.black)
                    c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - TOP_MARGIN - 60, "Key Metrics")


                    metric_lines = [line for line in content.split("\n") if ":" in line and "Last 30" not in line]
                    metrics = dict(line.split(":", 1) for line in metric_lines)
                    draw_metrics_grid(c, metrics, PAGE_HEIGHT - 180) 

                    # Page 2: Trend Heading & Paragraph
                    c.showPage()
                    if i < len(sections) - 1:
                        next_section = sections[i + 1]
                        adjust_page_height(c, next_section)

                    # next_section = sections[i + 1]
                    # adjust_page_height(c, next_section)

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
                            title_y = PAGE_HEIGHT - TOP_MARGIN - 60
                            c.drawCentredString(PAGE_WIDTH / 2, title_y, chart_title)

                            chart_width = PAGE_WIDTH - 1.5 * LEFT_MARGIN
                            chart_height = 350
                            chart_x = (PAGE_WIDTH - chart_width) / 2
                            chart_y = title_y - chart_height - 40 
                            #chart_y = max(BOTTOM_MARGIN + 40, title_y - chart_height - 30)
                            #chart_y = max(BOTTOM_MARGIN + 30, title_y - chart_height - 10)



                            img1 = ImageReader(charts[0][1])
                            c.drawImage(img1, chart_x, chart_y, width=chart_width, height=chart_height, preserveAspectRatio=True)


                        except Exception as e:
                            print(f"⚠️ Chart 1 render error: {str(e)}") 
                    #------------------------------------------------------------
                            
                    if len(charts) > 1:
                        c.showPage()
                        section = {"title": "3 Charts Section", "contains_table": False}
                        adjust_page_height(c, section)

                        
                        #PAGE_HEIGHT = 1400  # Increase to fit 3 charts
                        #LOGO_Y_OFFSET = PAGE_HEIGHT - TOP_MARGIN + 10
                        #c.setPageSize((PAGE_WIDTH, PAGE_HEIGHT))
                        draw_header(c)

                        chart_titles = [
                            "Purchases vs ROAS",
                            "CPA vs Link CPC",
                            "Click to Conversion vs CTR"
                        ]

                        chart_y = PAGE_HEIGHT - TOP_MARGIN - 60
                        chart_width = PAGE_WIDTH - 1.5 * LEFT_MARGIN
                        chart_height = 300
                        chart_spacing = 70  # space between charts

                        for idx, (title, chart_buf) in enumerate(charts[1:4]):
                            
                            try:
                                c.setFont("Helvetica-Bold", 14)
                                c.drawCentredString(PAGE_WIDTH / 2, chart_y, title)

                                img = ImageReader(chart_buf)
                                c.drawImage(
                                    img,
                                    (PAGE_WIDTH - chart_width) / 2,
                                    chart_y - chart_height - 10,
                                    width=chart_width,
                                    height=chart_height,
                                    preserveAspectRatio=True
                                )
                                chart_y -= chart_height + chart_spacing + 30
                            except Exception as e:
                                print(f"⚠️ Error rendering chart {title}: {str(e)}")
 
                
                    
                    # New Page: Full Table Summary
                    if ad_insights_df is not None and not ad_insights_df.empty:
                        # PAGE_HEIGHT = 1400
                        # LOGO_Y_OFFSET = PAGE_HEIGHT - TOP_MARGIN + 10
                        # c.setPageSize((PAGE_WIDTH, PAGE_HEIGHT))
                        c.showPage()
                       
                        table_section = {"title": "Campaign Performance Overview", "contains_table": True}
                        adjust_page_height(c, table_section)
                        
                
                        #draw_header(c)
                        #metric_top_y = PAGE_HEIGHT - TOP_MARGIN - 10
                        
                        #metric_lines = [line for line in content.split("\n") if ":" in line and "Last 30" not in line]
                        #metrics = dict(line.split(":", 1) for line in metric_lines)
                        #draw_metrics_grid(c, metrics, metric_top_y)
                        c.setFont("Helvetica-Bold", 18)
                        c.setFillColor(colors.black)
                        title_y = PAGE_HEIGHT - TOP_MARGIN - 20
                        c.drawCentredString(PAGE_WIDTH / 2, title_y, "Campaign Performance Overview")

                        
                        # title_y = PAGE_HEIGHT - TOP_MARGIN - 100
                        # c.setFont("Helvetica-Bold", 16)
                        # c.drawCentredString(PAGE_WIDTH / 2, title_y, "Campaign Performance Summary")
                        ad_insights_df = ad_insights_df.sort_values('date')
                        
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
                        # Prepare table data
                        table_data = [["Day", "Amount spent", "Purchases", "Purchases conversion value", "CPA", "Impressions","CTR", "Link clicks", "Click To Conversion", "ROAS"]]

                        import pandas as pd

                        for _, row in ad_insights_df.iterrows():
                            table_data.append([
                                pd.to_datetime(row['date']).strftime("%d %b %Y"),
                                f"{currency_symbol}{row['spend']:,.2f}",
                                int(row['purchases']),
                                f"{currency_symbol}{row['purchase_value']:,.2f}",
                                f"{currency_symbol}{row['cpa']:,.2f}",
                                f"{int(row['impressions']):,}",
                                f"{row['ctr']:.2%}",
                                int(row['clicks']),
                                f"{row['click_to_conversion']:.2%}",
                                f"{totals['roas']:.2f}",
                            ])

                        next_section["contains_table"] = True
                        next_section["table_rows"] = len(table_data)  # Or however many rows are going to be rendered

                        print("🖨 PDF row date:", row['date'], type(row['date']))

                        # Append grand total row
                        table_data.append([
                            "Grand Total",
                            f"{currency_symbol}{totals['spend']:,.2f}",
                            int(totals['purchases']),
                            f"{currency_symbol}{totals['purchase_value']:,.2f}",
                            f"{currency_symbol}{totals['cpa']:,.2f}",
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
                            #("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans" if currency_symbol == "₹" else "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, -1), 8),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("BACKGROUND", (0, -1), (-1, -1), colors.lightblue),  # Last row = Grand Total
                            #("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold")
                            #("FONTNAME", (0, -1), (-1, -1), "DejaVuSans" if currency_symbol == "₹" else "Helvetica-Bold"),
                        ]))

                        
                        #table_y = LOGO_Y_OFFSET - LOGO_HEIGHT - 20  
                        table_y = PAGE_HEIGHT - 1000  # You can adjust this to 400 if still too high
                        summary_table.wrapOn(c, PAGE_WIDTH, PAGE_HEIGHT)
                        summary_table.drawOn(c, LEFT_MARGIN, table_y)

                        #draw_footer = False  # Skip footer for table page

                    if full_ad_insights_df is not None and 'campaign_name' in full_ad_insights_df.columns:
                        c.showPage()
                        table_section = {"title": "Campaign Performance Summary", "contains_table": True}
                        adjust_page_height(c, table_section)  # ✅ this will now set PAGE_HEIGHT = 1800 automatically
                        

                        #table_section = {"title": "Campaign Performance Summary", "contains_table": True}
                        #adjust_page_height(c, table_section)
                        #c.showPage()
                       


                        draw_header(c)
                        df = full_ad_insights_df.copy()

                        c.setFont("Helvetica-Bold", 16)
                        c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - TOP_MARGIN - 30, "Campaign Level Performance")
                        df = full_ad_insights_df[full_ad_insights_df['campaign_name'].notna()]
                        if not df.empty:
                            df = full_ad_insights_df.copy()
                            df = df[df['campaign_name'].notna()]  # Filter out rows without campaign names
                            # Ensure numeric columns
                            numeric_cols = ['spend', 'purchase_value', 'purchases']
                            for col in numeric_cols:
                                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

                            # Group by campaign
                            grouped_campaigns = df.groupby('campaign_name').agg({
                                'spend': 'sum',
                                'purchase_value': 'sum',
                                'purchases': 'sum'
                            }).reset_index()

                            # Calculate metrics
                            grouped_campaigns['roas'] = grouped_campaigns.apply(
                                lambda row: row['purchase_value'] / row['spend'] if row['spend'] > 0 else 0, 
                                axis=1
                            )
                            grouped_campaigns['cpa'] = grouped_campaigns.apply(
                                lambda row: row['spend'] / row['purchases'] if row['purchases'] > 0 else 0, 
                                axis=1
                            )
                            

                            table_data = [["Campaign Name", "Amount Spent", "Revenue", "Purchases", "ROAS", "CPA"]]
                            for _, row in grouped_campaigns.iterrows():
                                table_data.append([
                                    row['campaign_name'],
                                    f"{currency_symbol}{row['spend']:,.2f}",
                                    f"{currency_symbol}{row['purchase_value']:,.2f}",
                                    int(row['purchases']),
                                    f"{row['roas']:.2f}",
                                    f"{currency_symbol}{row['cpa']:.2f}"
                                ])

                            next_section["contains_table"] = True
                            next_section["table_rows"] = len(table_data)

                            # Grand Total
                            total_spend = grouped_campaigns['spend'].sum()
                            total_purchases = grouped_campaigns['purchases'].sum()
                            total_purchase_value = grouped_campaigns['purchase_value'].sum()
                            grand_totals = {
                                'spend': total_spend,
                                'purchase_value': total_purchase_value,
                                'purchases': total_purchases,
                                'roas': total_purchase_value / total_spend if total_spend > 0 else 0,
                                'cpa': total_spend / total_purchases if total_purchases > 0 else 0
                            }

                            table_data.append([
                                "Grand Total",
                                f"{currency_symbol}{grand_totals['spend']:,.2f}",
                                f"{currency_symbol}{grand_totals['purchase_value']:,.2f}",
                                int(grand_totals['purchases']),
                                f"{grand_totals['roas']:.2f}",
                                f"{currency_symbol}{grand_totals['cpa']:.2f}"
                            ])

                            performance_table = Table(table_data, repeatRows=1, colWidths=[220, 110, 110, 90, 90, 90])
                            performance_table.setStyle(TableStyle([
                                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                                #("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans" if currency_symbol == "₹" else "Helvetica-Bold"),
                                ("FONTSIZE", (0, 0), (-1, -1), 8),
                                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                ("BACKGROUND", (0, -1), (-1, -1), colors.lightblue),
                                #("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold")
                                #("FONTNAME", (0, -1), (-1, -1), "DejaVuSans" if currency_symbol == "₹" else "Helvetica-Bold"),
                            ]))

                            table_y = PAGE_HEIGHT - TOP_MARGIN - 350
                            performance_table.wrapOn(c, PAGE_WIDTH, PAGE_HEIGHT)
                            performance_table.drawOn(c, LEFT_MARGIN, table_y)


                        else:
                            c.setFont("Helvetica", 12)
                            c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT / 2, 
                          "Campaign data available but no valid campaign names found")
                            
                        

                        # Draw Split Charts below the table
                        if 'split_charts' in locals() and split_charts and len(split_charts) >= 3:
                            # 🎯 All three charts on the same row inside a card
                            chart_width = 250
                            chart_height = 250
                            padding_x = 40

                            total_width = chart_width * 3 + padding_x * 2
                            start_x = (PAGE_WIDTH - total_width) / 2
                            chart_y = table_y - chart_height - 80

                            # Optional: Light gray card-style background
                            card_padding = 20
                            card_x = start_x - card_padding
                            card_y = chart_y - card_padding
                            card_w = total_width + 2 * card_padding
                            card_h = chart_height + 2 * card_padding

                            c.setFillColor(colors.whitesmoke)
                            c.roundRect(card_x, card_y, card_w, card_h, radius=12, fill=1, stroke=0)

                            # Chart 1 - Cost Split
                            if len(split_charts) > 0:
                                img1 = ImageReader(split_charts[0][1])
                                c.drawImage(img1, start_x, chart_y, width=chart_width, height=chart_height)

                            # Chart 2 - Revenue Split
                            if len(split_charts) > 1:
                                img2 = ImageReader(split_charts[1][1])
                                c.drawImage(img2, start_x + chart_width + padding_x, chart_y, width=chart_width, height=chart_height)

                            # Chart 3 - ROAS Split (horizontal bar)
                            if len(split_charts) > 2:
                                img3 = ImageReader(split_charts[2][1])
                                c.drawImage(img3, start_x + 2 * (chart_width + padding_x), chart_y, width=chart_width, height=chart_height)

                            
                            
                            try:
                                #c.showPage()
                                #PAGE_HEIGHT = 600
                                #TOP_MARGIN = 1.2 * inch
                                #LOGO_Y_OFFSET = PAGE_HEIGHT - TOP_MARGIN + 10
                                #c.setPageSize((PAGE_WIDTH, PAGE_HEIGHT))
                                #draw_header(c)

                                from services.deepseek_audit import generate_cost_by_campaign_chart  # Only if not already imported
                                cost_by_campaign_chart = generate_cost_by_campaign_chart(full_ad_insights_df)

                                # Draw title
                                #chart_title = "Cost by Campaigns"
                                #c.setFont("Helvetica-Bold", 16)
                                title_y = PAGE_HEIGHT - TOP_MARGIN - 60
                                #c.drawCentredString(PAGE_WIDTH / 2, title_y, chart_title)

                                # Draw chart image
                                img = ImageReader(cost_by_campaign_chart[1])
                                chart_width = PAGE_WIDTH - 1.5 * LEFT_MARGIN
                                chart_height = 420
                                chart_x = (PAGE_WIDTH - chart_width) / 2
                                #chart_y = BOTTOM_MARGIN + 100
                                chart_y = chart_y - chart_height - 40 
                                #chart_y = table_y - chart_height - 80

                                c.drawImage(img, chart_x, chart_y, width=chart_width, height=chart_height, preserveAspectRatio=True)

                            except Exception as e:
                                print(f"⚠️ Error rendering Cost by Campaigns chart: {str(e)}")
                                
                            # Draw Revenue by Campaigns on same page (below)
                            try:
                                from services.deepseek_audit import generate_revenue_by_campaign_chart

                                #chart_title = "Revenue by Campaigns"
                                #c.setFont("Helvetica-Bold", 16)
                                #revenue_chart_y = chart_y - chart_height - 50
                                revenue_chart_y = chart_y - 30
                                #c.drawCentredString(PAGE_WIDTH / 2, revenue_chart_y + 20, chart_title)

                                rev_chart = generate_revenue_by_campaign_chart(full_ad_insights_df)
                                rev_img = ImageReader(rev_chart[1])

                                revenue_chart_height = 420
                                revenue_chart_width = PAGE_WIDTH - 1.5 * LEFT_MARGIN
                                chart_x = (PAGE_WIDTH - revenue_chart_width) / 2
                                chart_y = revenue_chart_y - revenue_chart_height

                                c.drawImage(rev_img, chart_x, chart_y, width=revenue_chart_width, height=revenue_chart_height, preserveAspectRatio=True)

                            except Exception as e:
                                print(f"⚠️ Error rendering Revenue by Campaigns chart: {str(e)}")
                                
                            # LLM summary paragraph after Revenue by Campaigns
                            try:
                                from services.deepseek_audit import generate_roas_summary_text
                                import asyncio

                                summary_text = asyncio.run(generate_roas_summary_text(full_ad_insights_df, currency_symbol))
                                print("📄 LLM Summary Generated")

                                paragraph_lines = summary_text.strip().split("\n")
                                text_width = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
                                #c.setFont("Helvetica", 12)
                                set_font_with_currency(c, currency_symbol, size=12)
                                summary_y = chart_y - 20  

                                for line in paragraph_lines:
                                    #wrapped = simpleSplit(line.strip(), "Helvetica", 12, text_width)
                                    wrapped = simpleSplit(line.strip(), "DejaVuSans" if currency_symbol == "₹" else "Helvetica", 12, text_width)
                                    for wline in wrapped:
                                        c.drawString(LEFT_MARGIN, summary_y, wline)
                                        summary_y -= 14

                            except Exception as e:
                                print(f"⚠️ LLM Summary generation failed: {str(e)}")
                                
                            # PAGE_HEIGHT = 700
                            # LOGO_Y_OFFSET = PAGE_HEIGHT - TOP_MARGIN + 10
                            # c.setPageSize((PAGE_WIDTH, PAGE_HEIGHT))



                                
                    
                    else:
                        c.showPage()
                        if i < len(sections) - 1:
                            next_section = sections[i + 1]
                            adjust_page_height(c, next_section)

                        # next_section = sections[i + 1]
                        # adjust_page_height(c, next_section)            
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
                                    if i < len(sections) - 1:
                                        next_section = sections[i + 1]
                                        adjust_page_height(c, next_section)

                                    # next_section = sections[i + 1]
                                    # adjust_page_height(c, next_section)


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

                    if i < len(sections) - 1 and section_title.strip().upper() != "COST BY CAMPAIGNS":
                #if i < len(sections) - 1:

                        c.showPage()
                        if i < len(sections) - 1:
                            next_section = sections[i + 1]
                            adjust_page_height(c, next_section)

                        # next_section = sections[i + 1]
                        # adjust_page_height(c, next_section)



        c.save()
        buffer.seek(0)
        return StreamingResponse(io.BytesIO(buffer.read()), media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=audit_report.pdf"})

    except Exception as e:
        print(f"❌ Error generating PDF: {str(e)}")
        raise Exception(f"Failed to generate PDF: {str(e)}")
