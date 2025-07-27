from reportlab.lib.pagesizes import landscape
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.utils import simpleSplit
import io
import os
import numpy as np
import asyncio
import threading
from fastapi.responses import StreamingResponse
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Table, TableStyle
# In generate_pdf.py, update the imports at the top to ensure Table is imported:
from reportlab.platypus import Table, TableStyle  # Make sure this line is present and uncommented
from reportlab.lib import colors
import re
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
#from services.chart_utils import draw_donut_chart, generate_chart_image, draw_roas_split_bar_chart
from services.chart_utils import (
    draw_donut_chart,
    draw_roas_split_bar_chart,
    generate_chart_image,
    generate_cost_by_adset_chart,
    generate_campaign_split_charts,
    generate_revenue_by_adset_chart,
    generate_cost_split_by_age_chart,
    generate_cost_split_by_gender_chart,
    generate_revenue_split_by_age_chart,
    generate_revenue_split_by_gender_chart,
    generate_roas_split_by_age_chart,
    generate_roas_split_by_gender_chart
) # make sure it's imported at the top

from services.deepseek_audit import generate_llm_content, build_demographic_summary_prompt

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def run_async_in_thread(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
# def run_async_in_thread(coro):
#     result = {}
#     def runner():
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         result["value"] = loop.run_until_complete(coro)
#         loop.close()
#     t = threading.Thread(target=runner)
#     t.start()
#     t.join()
#     return result["value"]







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
        PAGE_HEIGHT = 800
    elif title == "CAMPAIGN PERFORMANCE SUMMARY":
        PAGE_HEIGHT = 2300
    elif title == "3 CHARTS SECTION":
        PAGE_HEIGHT = 1700
    elif title == "ADSET LEVEL PERFORMANCE":
        PAGE_HEIGHT = 2500
    elif title == "AD LEVEL PERFORMANCE":
        PAGE_HEIGHT = 3750 
    elif title == "AD FATIGUE ANALYSIS":
        PAGE_HEIGHT = 4000 
    elif title == "DEMOGRAPHIC PERFORMANCE":
        PAGE_HEIGHT = 1800   
    elif title == "PLATFORM LEVEL PERFORMANCE":
        PAGE_HEIGHT = 2000 
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
    

def run_async_in_thread(coro):
    result = {}
    def runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result["value"] = loop.run_until_complete(coro)
        loop.close()
    t = threading.Thread(target=runner)
    t.start()
    t.join()
    return result["value"]


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

        

def generate_pdf_report(sections: list, ad_insights_df=None,full_ad_insights_df=None, currency_symbol=None, split_charts=None,demographic_df=None, platform_df=None, date_since=None, date_until=None) -> StreamingResponse:
    global PAGE_HEIGHT, LOGO_Y_OFFSET, TOP_MARGIN  # ✅ Fixes UnboundLocalError

    if currency_symbol is None:
        currency_symbol = "₹"
        
    if 'actions' in full_ad_insights_df.columns:
        full_ad_insights_df.drop(columns=['actions'], inplace=True)
        
    # if ad_insights_df is not None and 'reach' not in ad_insights_df.columns:
    #     print("⚠️ 'reach' missing in ad_insights_df, creating fallback.")
    #     ad_insights_df['reach'] = ad_insights_df['impressions']
        
    # if full_ad_insights_df is not None and 'reach' not in full_ad_insights_df.columns:
    #     print("⚠️ 'reach' missing in full_ad_insights_df, creating fallback.")
    #     full_ad_insights_df['reach'] = full_ad_insights_df['impressions']
    
    if ad_insights_df is not None:
        if 'reach' not in ad_insights_df.columns and 'impressions' in ad_insights_df.columns:
            print("⚠️ 'reach' missing in ad_insights_df, using impressions as fallback")
            ad_insights_df['reach'] = ad_insights_df['impressions']
        elif 'reach' not in ad_insights_df.columns:
            print("⚠️ Both 'reach' and 'impressions' missing - setting reach to 1 to avoid division errors")
            ad_insights_df['reach'] = 1

    if full_ad_insights_df is not None:
        if 'reach' not in full_ad_insights_df.columns and 'impressions' in full_ad_insights_df.columns:
            print("⚠️ 'reach' missing in full_ad_insights_df, using impressions as fallback")
            full_ad_insights_df['reach'] = full_ad_insights_df['impressions']
        elif 'reach' not in full_ad_insights_df.columns:
            print("⚠️ Both 'reach' and 'impressions' missing - setting reach to 1 to avoid division errors")
            full_ad_insights_df['reach'] = 1

        
    # 🔥 Fallback: Ensure 'roas' exists
    if ad_insights_df is not None and 'roas' not in ad_insights_df.columns:
        print("⚠️ 'roas' missing in ad_insights_df, calculating fallback.")
        ad_insights_df['roas'] = ad_insights_df['purchase_value'] / ad_insights_df['spend'].replace(0, 1)
        ad_insights_df['roas'] = ad_insights_df['roas'].fillna(0).round(2)

    if full_ad_insights_df is not None and 'roas' not in full_ad_insights_df.columns:
        print("⚠️ 'roas' missing in full_ad_insights_df, calculating fallback.")
        full_ad_insights_df['roas'] = full_ad_insights_df['purchase_value'] / full_ad_insights_df['spend'].replace(0, 1)
        full_ad_insights_df['roas'] = full_ad_insights_df['roas'].fillna(0).round(2)



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
                # Draw divider on first 5 section pages only
            if section_title.upper() in ["EXECUTIVE SUMMARY","ACCOUNT NAMING & STRUCTURE","TESTING ACTIVITY","REMARKETING ACTIVITY","RESULTS SETUP"]:
                left_section_width = PAGE_WIDTH * 0.4
                text_x = left_section_width + 20
                c.setStrokeColor(colors.HexColor("#007bff"))
                c.setLineWidth(8)
                c.line(text_x - 10, BOTTOM_MARGIN, text_x - 10, PAGE_HEIGHT - TOP_MARGIN)
            

            if section_title.strip().upper() == "KEY METRICS":
                    # Debug print to verify currency symbol
                    print(f"🔎 Current currency symbol: {currency_symbol}")
                    # Page 1: Key Metrics Header & Cards
                    c.setFont("Helvetica-Bold", 24)
                    c.setFillColor(colors.black)
                    c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - TOP_MARGIN - 60, "Key Metrics")
                    
                    # 👉 Add this block for the subheading (just below main heading)
                    if date_since and date_until:
                        subheading = f"Insights date-range {date_since} to {date_until}"
                        c.setFont("Helvetica", 16)
                        c.setFillColor(colors.HexColor("#3B3B3B"))  # Light grey
                        c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - TOP_MARGIN - 100, subheading)


                    metric_lines = [line for line in content.split("\n") if ":" in line and "Last 30" not in line]
                    metrics = dict(line.split(":", 1) for line in metric_lines)
                    draw_metrics_grid(c, metrics, PAGE_HEIGHT - 220) 

                    # Page 2: Trend Heading & Paragraph
                    c.showPage()
                    draw_header(c)
                    c.setPageSize((PAGE_WIDTH, 600))
                    # if i < len(sections) - 1:
                    #     next_section = sections[i + 1]
                    #     adjust_page_height(c, next_section)
                    global LOGO_Y_OFFSET # Declare global if you are re-assigning it here
                    LOGO_Y_OFFSET = 700 - TOP_MARGIN + 10

                    # next_section = sections[i + 1]
                    # adjust_page_height(c, next_section)

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
                            #c.drawCentredString(PAGE_WIDTH / 2, title_y, chart_title)

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

                        chart_y = PAGE_HEIGHT - TOP_MARGIN - 80
                        #chart_width = PAGE_WIDTH - 1.5 * LEFT_MARGIN
                        chart_width = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
                        chart_height = 350
                        chart_spacing = 130  # space between charts

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
                                
                        # Draw footer CTA   
                        draw_footer_cta(c)
 
                
                    
                    # New Page: Full Table Summary
                    if ad_insights_df is not None and not ad_insights_df.empty:
                        # PAGE_HEIGHT = 1400
                        # LOGO_Y_OFFSET = PAGE_HEIGHT - TOP_MARGIN + 10
                        # c.setPageSize((PAGE_WIDTH, PAGE_HEIGHT))
                        c.showPage()
                       
                        table_section = {"title": "Campaign Performance Overview", "contains_table": True}
                        adjust_page_height(c, table_section)
                        
                
                        draw_header(c)
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
                                #f"{currency_symbol}{row['cpa']:,.2f}", NA
                                f"{currency_symbol}{row['cpa']:,.2f}" if pd.notna(row['cpa']) else "N/A",
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
                            #f"{currency_symbol}{totals['cpa']:,.2f}", NA
                            f"{currency_symbol}{totals['cpa']:,.2f}" if pd.notna(totals['cpa']) else "N/A",
                            f"{int(totals['impressions']):,}",
                            f"{totals['ctr']:.2%}",
                            int(totals['clicks']),
                            f"{totals['click_to_conversion']:.2%}",
                            f"{totals['roas']:.2f}",
                        ])

                        # Limit row count if needed (for fitting one page), or use page breaks
                        #summary_table = Table(table_data, repeatRows=1, colWidths=[90]*10)
                        summary_table = Table(table_data, repeatRows=1, colWidths=[90, 90, 90, 130, 80, 80,80,80,80,80])
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
                        table_y = PAGE_HEIGHT - 700  # You can adjust this to 400 if still too high
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
                            # Prepare ROAS data for campaign level
                            campaign_roas = grouped_campaigns.set_index('campaign_name')['roas']

                            

                            table_data = [["Campaign Name", "Amount Spent", "Revenue", "Purchases", "ROAS", "CPA"]]
                            for _, row in grouped_campaigns.iterrows():
                                table_data.append([
                                    row['campaign_name'],
                                    f"{currency_symbol}{row['spend']:,.2f}",
                                    f"{currency_symbol}{row['purchase_value']:,.2f}",
                                    int(row['purchases']),
                                    f"{row['roas']:.2f}",
                                    #f"{currency_symbol}{row['cpa']:.2f}" NA
                                    f"{currency_symbol}{row['cpa']:.2f}" if pd.notna(row['cpa']) else "N/A"
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
                                #f"{currency_symbol}{grand_totals['cpa']:.2f}" NA
                                f"{currency_symbol}{grand_totals['cpa']:.2f}" if pd.notna(grand_totals['cpa']) else "N/A"
                            ])

                            performance_table = Table(table_data, repeatRows=1, colWidths=[260, 140, 140, 100, 100, 100])
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

                            table_y = PAGE_HEIGHT - TOP_MARGIN - 200
                            performance_table.wrapOn(c, PAGE_WIDTH, PAGE_HEIGHT)
                            performance_table.drawOn(c, LEFT_MARGIN, table_y)


                        else:
                            c.setFont("Helvetica", 12)
                            c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT / 2, 
                          "Campaign data available but no valid campaign names found")
                            
                        
                        if 'split_charts' in locals() and split_charts and len(split_charts) >= 3:
                            chart_width = 350
                            chart_height = 350
                            padding_x = 40
                            padding_y = 40
                            
                            donut_width = 410
                            donut_height = 410
                            donut_y = table_y - donut_height - 40

                            # Cost Split - left aligned
                            cost_x = LEFT_MARGIN
                            c.setStrokeColor(colors.lightgrey)
                            c.setLineWidth(1)
                            c.roundRect(cost_x, donut_y, donut_width, donut_height, radius=8, fill=0, stroke=1)
                            if len(split_charts) > 0:
                                img1 = ImageReader(split_charts[0][1])
                                c.drawImage(img1, cost_x, donut_y, width=donut_width, height=donut_height)

                            # Revenue Split - right aligned
                            revenue_x = PAGE_WIDTH - RIGHT_MARGIN - donut_width
                            c.setStrokeColor(colors.lightgrey)
                            c.setLineWidth(1)
                            c.roundRect(revenue_x, donut_y, donut_width, donut_height, radius=8, fill=0, stroke=1)
                            if len(split_charts) > 1:
                                img2 = ImageReader(split_charts[1][1])
                                c.drawImage(img2, revenue_x, donut_y, width=donut_width, height=donut_height)


                            # Calculate position for top row (2 donut charts)
                            total_width = chart_width * 2 + padding_x
                            start_x = (PAGE_WIDTH - total_width) / 2
                            top_chart_y = table_y - chart_height - 40

                        
 
                        # ────────────── Chart 3: ROAS Split ──────────────
                            roas_width = 740
                            roas_height = 320
                            roas_x = (PAGE_WIDTH - roas_width) / 2
                            #roas_y = top_chart_y - roas_height - 60
                            roas_y = donut_y - 40 - roas_height  # ensures enough gap


                            # Title
                            c.setFont("Helvetica-Bold", 13)
                            c.setFillColor(colors.black)
                            #c.drawCentredString(PAGE_WIDTH / 2, roas_y + roas_height + 16, "ROAS Split")

                            # Card Border
                            c.setStrokeColor(colors.lightgrey)
                            c.setLineWidth(1)
                            c.roundRect(roas_x, roas_y, roas_width, roas_height, radius=8, fill=0, stroke=1)

                            # if len(split_charts) > 2:
                            #     img3 = ImageReader(split_charts[2][1])
                            #     c.drawImage(img3, roas_x, roas_y, width=roas_width, height=roas_height)
                            from services.chart_utils import roas_split_campaign
                            try:
                                chart_title, chart_img = roas_split_campaign(campaign_roas)
                                img3 = ImageReader(chart_img)
                                c.drawImage(img3, roas_x, roas_y, width=roas_width, height=roas_height)
                            except Exception as e:
                                print(f"⚠️ Error rendering ROAS Split Campaign chart: {e}")

                            
                            try:
                               

                                from services.chart_utils import generate_cost_by_campaign_chart  # Only if not already imported
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
                                #chart_y = chart_y - chart_height - 40 
                                chart_y = roas_y - 60 - chart_height  # place below ROAS chart cleanly
                                #chart_y = table_y - chart_height - 80

                                c.drawImage(img, chart_x, chart_y, width=chart_width, height=chart_height, preserveAspectRatio=True)

                            except Exception as e:
                                print(f"⚠️ Error rendering Cost by Campaigns chart: {str(e)}")
                                
                            # Draw Revenue by Campaigns on same page (below)
                            try:
                                from services.chart_utils import generate_revenue_by_campaign_chart

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

                                summary_text = run_async_in_thread(generate_roas_summary_text(full_ad_insights_df, currency_symbol))

                                print("📄 LLM Summary Generated")

                                paragraph_lines = summary_text.strip().split("\n")
                                text_width = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
                                #c.setFont("Helvetica", 12)
                                set_font_with_currency(c, currency_symbol, size=12)
                                c.setFillColor(colors.black) 
                                summary_y = chart_y - 70  

                                for line in paragraph_lines:
                                    #wrapped = simpleSplit(line.strip(), "Helvetica", 12, text_width)
                                    wrapped = simpleSplit(line.strip(), "DejaVuSans" if currency_symbol == "₹" else "Helvetica", 12, text_width)
                                    for wline in wrapped:
                                        c.drawString(LEFT_MARGIN, summary_y, wline)
                                        summary_y -= 14
                                        
                                draw_footer_cta(c)  # Draw footer CTA after LLM summary

                            except Exception as e:
                                print(f"⚠️ LLM Summary generation failed: {str(e)}")
                                
                            # PAGE_HEIGHT = 700
                            # LOGO_Y_OFFSET = PAGE_HEIGHT - TOP_MARGIN + 10
                            # c.setPageSize((PAGE_WIDTH, PAGE_HEIGHT))
                            
                            
                            #New Page - Adset Level Performance--------------------------------------------
                            c.showPage()
                            adjust_page_height(c, {"title": "Adset Level Performance", "contains_table": True})
                            draw_header(c)
                            
                            # ✅ Add title
                            c.setFont("Helvetica-Bold", 20)
                            c.setFillColor(colors.black)
                            c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - TOP_MARGIN - 30, "Adset Level Performance")
                            
                            df = full_ad_insights_df.copy()
                            df = df[df['adset_name'].notna()]
                            df['spend'] = pd.to_numeric(df['spend'], errors='coerce').fillna(0)
                            df['purchase_value'] = pd.to_numeric(df['purchase_value'], errors='coerce').fillna(0)
                            df['purchases'] = pd.to_numeric(df['purchases'], errors='coerce').fillna(0)
                            df['roas'] = df['purchase_value'] / df['spend'].replace(0, 1)
                            df['cpa'] = df['spend'] / df['purchases'].replace(0, 1)

                            grouped = df.groupby('adset_name').agg({
                                'spend': 'sum',
                                'purchase_value': 'sum',
                                'purchases': 'sum',
                                'roas': 'mean',
                                'cpa': 'mean'
                            }).reset_index()
                            
                            # Prepare top 6 adsets by spend, revenue, and ROAS
                            top_spend = grouped.set_index('adset_name')['spend'].sort_values(ascending=False).head(6)
                            top_revenue = grouped.set_index('adset_name')['purchase_value'].sort_values(ascending=False).head(6)
                            top_roas = grouped.set_index('adset_name')['roas'].sort_values(ascending=False).head(6)


                            table_data = [["Adset Name", "Amount Spent", "Revenue", "Purchases", "ROAS", "CPA"]]
                            for _, row in grouped.iterrows():
                                table_data.append([
                                    
                                    row['adset_name'],
                                    f"{currency_symbol}{row['spend']:.2f}",
                                    f"{currency_symbol}{row['purchase_value']:.2f}",
                                    int(row['purchases']),
                                    f"{row['roas']:.2f}" if pd.notna(row['roas']) else "N/A",
                                    #f"{currency_symbol}{row['cpa']:.2f}" NA
                                    f"{currency_symbol}{row['cpa']:.2f}" if pd.notna(row['cpa']) else "N/A"
                                ])
                                
                            # ➤ Grand Total Calculation
                            total_spend = grouped['spend'].sum()
                            total_revenue = grouped['purchase_value'].sum()
                            total_purchases = grouped['purchases'].sum()

                            valid_roas = grouped['roas'].dropna()
                            valid_cpa = grouped['cpa'].dropna()

                            total_roas = valid_roas.mean() if not valid_roas.empty else None
                            total_cpa = valid_cpa.mean() if not valid_cpa.empty else None

                            # ➤ Append Grand Total Row
                            table_data.append([
                                "Grand Total",
                                f"{currency_symbol}{total_spend:.2f}",
                                f"{currency_symbol}{total_revenue:.2f}",
                                int(total_purchases),
                                f"{total_roas:.2f}" if total_roas is not None else "N/A",
                                f"{currency_symbol}{total_cpa:.2f}" if total_cpa is not None else "N/A"
                            ])

                            summary_table = Table(table_data, repeatRows=1, colWidths=[270, 130, 130, 90, 90, 110])
                            summary_table.setStyle(TableStyle([
                                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                                ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans"),
                                ("FONTSIZE", (0, 0), (-1, -1), 8),
                                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                ("BACKGROUND", (0, -1), (-1, -1), colors.lightblue),
                            ]))
                            table_y = PAGE_HEIGHT - TOP_MARGIN - 300
                            summary_table.wrapOn(c, PAGE_WIDTH, PAGE_HEIGHT)
                            summary_table.drawOn(c, LEFT_MARGIN, table_y)
                            
                            # ─────────────────────────────
                            # 🎯 Donut + ROAS Split Section
                            # ─────────────────────────────
                            # ─────────────── Donut Charts (Left + Right Aligned) ───────────────
                            donut_width = 380
                            donut_height = 380
                            large_chart_height = 480
                            donut_padding_y = 40
                            #donut_y = table_y - donut_height - donut_padding_y
                            donut_y = table_y - donut_height - 40

                            # Cost Split – flush left
                            cost_x = LEFT_MARGIN
                            c.setStrokeColor(colors.lightgrey)
                            c.setLineWidth(1)
                            c.roundRect(cost_x, donut_y, donut_width, donut_height, radius=8, fill=0, stroke=1)
                            print("📊 top_spend:", top_spend)
                            print("📊 top_revenue:", top_revenue)
                            print("📊 top_roas:", top_roas)


                            try:
                                fig1 = draw_donut_chart(top_spend.values, top_spend.index, "Cost Split")
                                img1 = ImageReader(generate_chart_image(fig1))
                                c.drawImage(img1, cost_x, donut_y, width=donut_width, height=donut_height)
                            except Exception as e:
                                c.setFont("Helvetica", 10)
                                c.setFillColor(colors.red)
                                c.drawString(cost_x + 20, donut_y + donut_height / 2, f"⚠️ Cost Split chart failed: {str(e)}")

                            # Revenue Split – flush right
                            revenue_x = PAGE_WIDTH - RIGHT_MARGIN - donut_width
                            c.setStrokeColor(colors.lightgrey)
                            c.setLineWidth(1)
                            c.roundRect(revenue_x, donut_y, donut_width, donut_height, radius=8, fill=0, stroke=1)
                            print("🎯 ROAS values:", top_roas.to_dict())

                            try:
                                fig2 = draw_donut_chart(top_revenue.values, top_revenue.index, "Revenue Split")
                                img2 = ImageReader(generate_chart_image(fig2))
                                c.drawImage(img2, revenue_x, donut_y, width=donut_width, height=donut_height)
                            except Exception as e:
                                print(f"⚠️ Error rendering Revenue Split: {str(e)}")


                            # Row 2: ROAS Bar Chart (Center with Heading)
                            roas_width = 770
                            roas_height = 280
                            roas_x = (PAGE_WIDTH - roas_width) / 2
                            #roas_y = top_chart_y - roas_height - 40
                            roas_y = donut_y - roas_height - 40 


                            # Heading above ROAS chart
                            c.setFont("Helvetica-Bold", 13)
                            c.setFillColor(colors.black)
                            #c.drawCentredString(PAGE_WIDTH / 2, roas_y + roas_height + 16, "ROAS Split")
 
                            # Card
                            c.setStrokeColor(colors.lightgrey)
                            c.setLineWidth(1)
                            c.roundRect(roas_x, roas_y, roas_width, roas_height, radius=8, fill=0, stroke=1)
                            try:
                                fig3 = draw_roas_split_bar_chart(top_roas)
                                img3 = ImageReader(generate_chart_image(fig3))
                                c.drawImage(img3, roas_x, roas_y, width=roas_width, height=roas_height)
                            except Exception as e:
                                print(f"⚠️ Error rendering ROAS Split: {str(e)}")
                                
                            from services.chart_utils import generate_cost_by_adset_chart, generate_revenue_by_adset_chart
                            card_width = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN + 20
                            chart_x = (PAGE_WIDTH - card_width) / 2

                                
                            #Cost by Adsets chart
                            cost_chart = generate_cost_by_adset_chart(full_ad_insights_df)
                            img1 = ImageReader(cost_chart[1])
                            c.setFont("Helvetica-Bold", 14)
                            c.setFillColor(colors.black)
                            cost_chart_y = roas_y - large_chart_height - 60
                            #c.drawCentredString(PAGE_WIDTH / 2, cost_chart_y + large_chart_height + 20, "Cost by Adsets")
                            c.drawImage(img1, chart_x + 20, cost_chart_y,
                            width=card_width - 40, height=large_chart_height)

                            #Revenue by Adsets chart
                            revenue_chart = generate_revenue_by_adset_chart(full_ad_insights_df)
                            img2 = ImageReader(revenue_chart[1])
                            c.setFont("Helvetica-Bold", 14)
                            revenue_chart_y = cost_chart_y - large_chart_height - 60
                            #c.drawCentredString(PAGE_WIDTH / 2, revenue_chart_y + large_chart_height + 20, "Revenue by Adsets")
                            c.drawImage(img2, chart_x + 20, revenue_chart_y,
                            width=card_width - 40, height=large_chart_height)


                           
                            try:                              

                                from services.deepseek_audit import generate_adset_summary

                                summary_text = run_async_in_thread(generate_adset_summary(full_ad_insights_df, currency_symbol))
                                print("📄 LLM Summary Generated")

                                paragraph_lines = summary_text.strip().split("\n")
                                text_width = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
                                #c.setFont("Helvetica", 12)
                                set_font_with_currency(c, currency_symbol, size=12)
                                c.setFillColor(colors.black) 
                                summary_y = chart_y - 80  

                                import re

                                # Clean the summary text: remove #, *, extra spaces
                                clean_text = re.sub(r"[*#]", "", summary_text).strip()
                                clean_text = re.sub(r"\s{2,}", " ", clean_text)  # Replace multiple spaces with one

                                # Move summary further down (below both charts)
                                #summary_y = chart_y - chart_height - 500
                                summary_y = revenue_chart_y - 60

                                # Set font and color
                                set_font_with_currency(c, currency_symbol, size=12)
                                c.setFillColor(colors.HexColor("#333333"))

                                # Wrap text for PDF width
                                from reportlab.platypus import Paragraph    
                                from reportlab.lib.styles import getSampleStyleSheet

                                styles = getSampleStyleSheet()
                                styleN = styles["Normal"]
                                styleN.fontName = "DejaVuSans" if currency_symbol == "₹" else "Helvetica"
                                styleN.fontSize = 11
                                styleN.leading = 14
                                styleN.textColor = colors.HexColor("#333333")

                                # Draw as paragraph
                                p = Paragraph(clean_text, styleN)
                                p_width, p_height = p.wrap(PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN, PAGE_HEIGHT)
                                p.drawOn(c, LEFT_MARGIN, summary_y - p_height)

                                        
                                draw_footer_cta(c)  # Draw footer CTA after LLM summary

                            except Exception as e:
                                print(f"⚠️ LLM Summary generation failed: {str(e)}")  
                                
                            
                            
                            # Add this after "Adset Level Performance" page block in generate_pdf.py

                            # ───────────────────────────────
                            # 📄 New Page - Ad Level Performance
                            # ───────────────────────────────
                            c.showPage()
                            adset_section = {"title": "Ad Level Performance", "contains_table": True}
                            adjust_page_height(c, adset_section)
                            draw_header(c)

                            # ➤ Prepare Ad-level data
                            ad_df = full_ad_insights_df.copy()
                            ad_df = ad_df[ad_df['ad_name'].notna()]
                            ad_df['spend'] = pd.to_numeric(ad_df['spend'], errors='coerce').fillna(0)
                            ad_df['purchase_value'] = pd.to_numeric(ad_df['purchase_value'], errors='coerce').fillna(0)
                            ad_df['purchases'] = pd.to_numeric(ad_df['purchases'], errors='coerce').fillna(0)
                            ad_df['roas'] = ad_df['purchase_value'] / ad_df['spend'].replace(0, 1)
                            ad_df['cpa'] = ad_df['spend'] / ad_df['purchases'].replace(0, 1)

                            # ➤ Aggregate by ad
                            ad_grouped = ad_df.groupby('ad_name').agg({
                                'spend': 'sum',
                                'purchase_value': 'sum',
                                'purchases': 'sum',
                                'roas': 'mean',
                                'cpa': 'mean'
                            }).reset_index()

                            #   ➤ Table Title
                            c.setFont("Helvetica-Bold", 20)
                            c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - TOP_MARGIN - 40, "Ad Level Performance")

                            # ➤ Table
                            ad_table_data = [["Ad Name", "Amount Spent", "Revenue", "Purchases", "ROAS", "CPA"]]
                            for _, row in ad_grouped.iterrows():
                                ad_table_data.append([
                                row['ad_name'],
                                f"{currency_symbol}{row['spend']:.2f}",
                                f"{currency_symbol}{row['purchase_value']:.2f}",
                                int(row['purchases']),
                                f"{row['roas']:.2f}"  if pd.notna(row['roas']) else "N/A",
                                #f"{currency_symbol}{row['cpa']:.2f}" NA
                                f"{currency_symbol}{row['cpa']:.2f}" if pd.notna(row['cpa']) else "N/A"
                            ])
                            # ➤ Compute Grand Totals
                            total_spend = ad_grouped['spend'].sum()
                            total_revenue = ad_grouped['purchase_value'].sum()
                            total_purchases = ad_grouped['purchases'].sum()

                            # Safe ROAS and CPA calculations
                            valid_roas = ad_grouped['roas'].dropna()
                            valid_cpa = ad_grouped['cpa'].dropna()

                            total_roas = valid_roas.mean() if not valid_roas.empty else None
                            total_cpa = valid_cpa.mean() if not valid_cpa.empty else None
                            
                            # ➤ Append Grand Total row
                            ad_table_data.append([
                                "Grand Total",
                                f"{currency_symbol}{total_spend:.2f}",
                                f"{currency_symbol}{total_revenue:.2f}",
                                int(total_purchases),
                                f"{total_roas:.2f}" if total_roas is not None else "N/A",
                                f"{currency_symbol}{total_cpa:.2f}" if total_cpa is not None else "N/A"
                            ])

                            ad_summary_table = Table(ad_table_data, repeatRows=1, colWidths=[250, 130, 130, 80, 90, 120])
                            ad_summary_table.setStyle(TableStyle([
                                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                                ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans"),
                                ("FONTSIZE", (0, 0), (-1, -1), 8),
                                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                ("BACKGROUND", (0, -1), (-1, -1), colors.lightblue),
                            ]))
                            ad_table_y = PAGE_HEIGHT - TOP_MARGIN - 1250
                            ad_summary_table.wrapOn(c, PAGE_WIDTH, PAGE_HEIGHT)
                            ad_summary_table.drawOn(c, LEFT_MARGIN, ad_table_y)

                            # ➤ Charts
                            from services.chart_utils import generate_cost_by_adset_chart, generate_revenue_by_adset_chart

                            # Top 6 ads
                            top_ad_spend = ad_grouped.set_index('ad_name')['spend'].sort_values(ascending=False).head(6)
                            top_ad_revenue = ad_grouped.set_index('ad_name')['purchase_value'].sort_values(ascending=False).head(6)
                            top_ad_roas = ad_grouped.set_index('ad_name')['roas'].sort_values(ascending=False).head(6)

                            # Donut Charts
                            donut_width, donut_height = 400, 400
                            donut_y = ad_table_y - donut_height - 40
                            padding_inner = 20

                            # Cost Split (left)
                            cost_x = LEFT_MARGIN
                            c.setStrokeColor(colors.lightgrey)  # Replace pink with grey
                            c.setLineWidth(1)
                            c.roundRect(cost_x, donut_y, donut_width, donut_height, radius=8, fill=0, stroke=1)
                            fig1 = draw_donut_chart(top_ad_spend.values, top_ad_spend.index, "")
                            #c.drawImage(ImageReader(generate_chart_image(fig1)), cost_x, donut_y, width=donut_width, height=donut_height)
                            c.drawImage(
                                ImageReader(generate_chart_image(fig1)),
                                cost_x + padding_inner / 2,                   # Shift right a bit
                                donut_y + padding_inner / 2,                  # Shift up slightly if needed
                                width=donut_width - padding_inner,            # Reduce width to create padding
                                height=donut_height - padding_inner           # Optional: reduce height too
                            )

                            # Revenue Split (right)
                            revenue_x = PAGE_WIDTH - RIGHT_MARGIN - donut_width
                            c.setStrokeColor(colors.lightgrey)  # Replace pink with grey
                            c.setLineWidth(1)
                            c.roundRect(revenue_x, donut_y, donut_width, donut_height, radius=8, fill=0, stroke=1)
                            fig2 = draw_donut_chart(top_ad_revenue.values, top_ad_revenue.index, "")
                            #c.drawImage(ImageReader(generate_chart_image(fig2)), revenue_x, donut_y, width=donut_width, height=donut_height)
                            c.drawImage(
                                ImageReader(generate_chart_image(fig2)),
                                revenue_x + padding_inner / 2,             # Correct X position
                                donut_y + padding_inner / 2,                 # Shift up slightly if needed
                                width=donut_width - padding_inner,            # Reduce width to create padding
                                height=donut_height - padding_inner           # Optional: reduce height too
                            )

                            # ROAS Split bar (centered)
                            roas_width, roas_height = 700, 300
                            roas_x = (PAGE_WIDTH - roas_width) / 2
                            roas_y = donut_y - roas_height - 60
                            c.roundRect(roas_x, roas_y, roas_width, roas_height, radius=8, fill=0, stroke=1)
                            fig3 = draw_roas_split_bar_chart(top_ad_roas)
                            c.drawImage(ImageReader(generate_chart_image(fig3)), roas_x, roas_y, width=roas_width, height=roas_height)

                            # Line charts
                            cost_chart = generate_cost_by_adset_chart(ad_df)
                            cost_chart_y = roas_y - 480 - 60
                            #c.drawCentredString(PAGE_WIDTH / 2, cost_chart_y + 480 + 20, "Cost by Ads")
                            c.drawImage(ImageReader(cost_chart[1]), LEFT_MARGIN, cost_chart_y, width=PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN, height=480)

                            revenue_chart = generate_revenue_by_adset_chart(ad_df)
                            revenue_chart_y = cost_chart_y - 480 - 60
                            #c.drawCentredString(PAGE_WIDTH / 2, revenue_chart_y + 480 + 20, "Revenue by Ads")
                            c.drawImage(ImageReader(revenue_chart[1]), LEFT_MARGIN, revenue_chart_y, width=PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN, height=480)

                            # ➤ Ad Summary (LLM generated — inline on same page)
                            try:
                                from services.deepseek_audit import generate_ad_summary

                                summary_text = run_async_in_thread(generate_ad_summary(ad_df, currency_symbol))

                                print("📄 LLM Summary Generated")

                                paragraph_lines = summary_text.strip().split("\n")
                                text_width = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
                                #c.setFont("Helvetica", 12)
                                set_font_with_currency(c, currency_symbol, size=12)
                                c.setFillColor(colors.black) 
                                summary_y = revenue_chart_y - 80  

                                import re

                                # Clean the summary text: remove #, *, extra spaces
                                clean_text = re.sub(r"[*#]", "", summary_text).strip()
                                clean_text = re.sub(r"\s{2,}", " ", clean_text)  # Replace multiple spaces with one

                                # Move summary further down (below both charts)
                                #summary_y = chart_y - chart_height - 500
                                summary_y = revenue_chart_y - 60

                                # Set font and color
                                set_font_with_currency(c, currency_symbol, size=12)
                                c.setFillColor(colors.HexColor("#333333"))

                                # Wrap text for PDF width
                                from reportlab.platypus import Paragraph    
                                from reportlab.lib.styles import getSampleStyleSheet

                                styles = getSampleStyleSheet()
                                styleN = styles["Normal"]
                                styleN.fontName = "DejaVuSans" if currency_symbol == "₹" else "Helvetica"
                                styleN.fontSize = 11
                                styleN.leading = 14
                                styleN.textColor = colors.HexColor("#333333")

                                # Draw as paragraph
                                p = Paragraph(clean_text, styleN)
                                p_width, p_height = p.wrap(PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN, PAGE_HEIGHT)
                                p.drawOn(c, LEFT_MARGIN, summary_y - p_height)
                                
                                draw_footer_cta(c)  # Draw footer CTA after LLM summary

                                        
                                
                            except Exception as e:
                                print(f"⚠️ LLM Summary generation failed: {str(e)}")
                                
                            c.showPage()
                            # ────────────── New Page: Ad Fatigue Analysis ──────────────
                            adjust_page_height(c, {"title": "Ad Fatigue Analysis", "contains_table": True})
                            draw_header(c)
                            MAX_FATIGUE_TABLE_ROWS = 100

                            # ✅ Add title
                            c.setFont("Helvetica-Bold", 20)
                            c.setFillColor(colors.black)
                            c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - TOP_MARGIN - 30, "Ad Fatigue Analysis")

                            # ➤ Prepare table data
                            df = full_ad_insights_df.copy()
                            # Sort by a relevant metric (e.g., spend, frequency) before limiting, to show most relevant ads
                            # For ad fatigue, sorting by frequency descending might be useful.
                            df['frequency'] = df['impressions'] / df['reach'].replace(0, 1) # Ensure frequency is calculated before sorting on it
                            df['cpa'] = df.apply(lambda row: row['spend'] / row['purchases'] if row['purchases'] > 0 else np.nan, axis=1)

                            # Sort by frequency (descending) or another relevant metric (e.g., spend)
                            # Then take the top N rows using .head()
                            df = df.sort_values(by='frequency', ascending=False).head(MAX_FATIGUE_TABLE_ROWS)
                            # --- END MODIFICATION ---
                            logger.info(f"📊 Columns in DataFrame: {list(df.columns)}")
                            logger.info(f"📊 First 5 rows:\n{df.head(5).to_string()}")
                            df['frequency'] = df['impressions'] / df['reach'].replace(0, 1)
                            table_data = [["Ad Name", "Campaign Name", "Adset Name", "Amount Spent", "Impressions", "Frequency", "ROAS", "CTR", "Purchases", "Purchase Conversion Value"]]
                            for _, row in df.iterrows():
                                table_data.append([
                                    row['ad_name'],
                                    row['campaign_name'],
                                    row['adset_name'],
                                    f"{currency_symbol}{row['spend']:.2f}",
                                    int(row['impressions']),
                                    f"{row['frequency']:.2f}",
                                    f"{row['roas']:.2f}",
                                    f"{row['ctr']:.2%}",
                                    int(row['purchases']),
                                    f"{currency_symbol}{row['purchase_value']:.2f}"
                                ])
                                
                            # ➤ Grand Total Calculations
                            total_spend = df['spend'].sum()
                            total_impressions = df['impressions'].sum()
                            total_purchases = df['purchases'].sum()
                            total_purchase_value = df['purchase_value'].sum()

                            valid_frequencies = df['frequency'].dropna()
                            valid_roas = df['roas'].dropna()
                            valid_ctr = df['ctr'].dropna()

                            total_frequency = valid_frequencies.mean() if not valid_frequencies.empty else None
                            total_roas = valid_roas.mean() if not valid_roas.empty else None
                            total_ctr = valid_ctr.mean() if not valid_ctr.empty else None

                            # ➤ Append Grand Total row
                            table_data.append([
                                "Grand Total",
                                "",  # campaign
                                "",  # adset
                                f"{currency_symbol}{total_spend:.2f}",
                                int(total_impressions),
                                f"{total_frequency:.2f}" if total_frequency is not None else "N/A",
                                f"{total_roas:.2f}" if total_roas is not None else "N/A",
                                f"{total_ctr:.2%}" if total_ctr is not None else "N/A",
                                int(total_purchases),
                                f"{currency_symbol}{total_purchase_value:.2f}"
                            ])

                            summary_table = Table(table_data, repeatRows=1, colWidths=[150, 150, 170, 70, 50, 60, 60, 60, 40, 60])
                            summary_table.setStyle(TableStyle([
                                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                                ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans" if currency_symbol == "₹" else "Helvetica-Bold"),
                                ("FONTSIZE", (0, 0), (-1, -1), 8),
                                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                ("BACKGROUND", (0, -1), (-1, -1), colors.lightblue),
                            ]))
                            
                            # Prepare ROAS series for Ad Fatigue
                            ad_fatigue_roas = df.groupby('ad_name')['roas'].mean().sort_values(ascending=False)

                            table_y = PAGE_HEIGHT - TOP_MARGIN - 1900
                            summary_table.wrapOn(c, PAGE_WIDTH, PAGE_HEIGHT)
                            summary_table.drawOn(c, LEFT_MARGIN, table_y)

                            # ────────────── Donut Charts ──────────────
                            donut_y = table_y - 450  # Adjust spacing
                            donut_width = 410
                            donut_height = 410

                            # # Cost Split
                            # cost_split = generate_campaign_split_charts(df, currency_symbol)[0][1]
                            # img_cost = ImageReader(cost_split)
                            # c.drawImage(img_cost, LEFT_MARGIN, donut_y, width=donut_width, height=donut_height)

                            # # Revenue Split
                            # revenue_split = generate_campaign_split_charts(df, currency_symbol)[1][1]
                            # img_revenue = ImageReader(revenue_split)
                            # c.drawImage(img_revenue, PAGE_WIDTH - RIGHT_MARGIN - donut_width, donut_y, width=donut_width, height=donut_height)
                            
                            from services.chart_utils import ad_fatigue_cost_donut, ad_fatigue_revenue_donut

                            # Cost Split (Ad Fatigue)
                            chart_title, cost_img = ad_fatigue_cost_donut(df)
                            img_cost = ImageReader(cost_img)
                            c.drawImage(img_cost, LEFT_MARGIN, donut_y, width=donut_width, height=donut_height)

                            # Revenue Split (Ad Fatigue)
                            chart_title, revenue_img = ad_fatigue_revenue_donut(df)
                            img_revenue = ImageReader(revenue_img)
                            c.drawImage(img_revenue, PAGE_WIDTH - RIGHT_MARGIN - donut_width, donut_y, width=donut_width, height=donut_height)


                            # ────────────── ROAS Split Bar Chart ──────────────
                            # roas_chart = generate_campaign_split_charts(df, currency_symbol)[2][1]
                            # img_roas = ImageReader(roas_chart)
                            # roas_y = donut_y - 300
                            # roas_width = 740
                            # roas_height = 320
                            # c.drawImage(img_roas, (PAGE_WIDTH - roas_width) / 2, roas_y, width=roas_width, height=roas_height)
                            
                            # ────────────── ROAS Split Bar Chart (New Design) ──────────────
                            from services.chart_utils import roas_split_Ad_Fatigue

                            try:
                                chart_title, chart_img = roas_split_Ad_Fatigue(ad_fatigue_roas)
                                img_roas = ImageReader(chart_img)
                                roas_y = donut_y - 300
                                roas_width = 740
                                roas_height = 320
                                c.drawImage(img_roas, (PAGE_WIDTH - roas_width) / 2, roas_y, width=roas_width, height=roas_height)
                            except Exception as e:
                                print(f"⚠️ Error rendering ROAS Split Ad Fatigue chart: {e}")


                            # ────────────── Frequency Over Time Chart ──────────────
                            from services.chart_utils import generate_frequency_over_time_chart
                            freq_chart = generate_frequency_over_time_chart(df)
                            img_freq = ImageReader(freq_chart[1])
                            freq_y = roas_y - 470
                            c.drawImage(img_freq, LEFT_MARGIN, freq_y, width=PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN, height=420)

                            # ────────────── CPM Over Time Chart ──────────────
                            from services.chart_utils import generate_cpm_over_time_chart
                            cpm_chart = generate_cpm_over_time_chart(df)
                            img_cpm = ImageReader(cpm_chart[1])
                            cpm_y = freq_y - 470
                            c.drawImage(img_cpm, LEFT_MARGIN, cpm_y, width=PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN, height=420)  
                            
                            draw_footer_cta(c)  # Draw footer CTA after Ad Fatigue section
                            
                            
                            # 📄 New Page - Demographic Performance
                            c.showPage()
                            adjust_page_height(c, {"title": "Demographic Performance", "contains_table": True})
                            #draw_header(c)

                            c.setFont("Helvetica-Bold", 20)
                            c.setFillColor(colors.black)
                            c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - TOP_MARGIN - 20, "Demographic Performance")

                            # ✅ Check for valid demographic data *before* attempting to process it
                            if demographic_df is not None and not demographic_df.empty and \
                                'age' in demographic_df.columns and 'gender' in demographic_df.columns and \
                                demographic_df['spend'].sum() > 0: # Ensure there's some spend data too

                                logger.info("Proceeding with Demographic Performance section as data is valid.")

                                # Ensure all required columns for aggregation exist and are numeric
                                for col in ['spend', 'purchase_value', 'purchases']:
                                    if col not in demographic_df.columns:
                                        demographic_df[col] = 0 # Add missing column with default 0
                                    else:
                                        demographic_df[col] = pd.to_numeric(demographic_df[col], errors='coerce').fillna(0)

                                 # Recalculate ROAS and CPA after ensuring numeric columns
                                demographic_df['roas'] = demographic_df['purchase_value'] / demographic_df['spend'].replace(0, 1)
                                demographic_df['cpa'] = demographic_df['spend'] / demographic_df['purchases'].replace(0, 1)

                                demographic_grouped = demographic_df.groupby(['age', 'gender']).agg({
                                    'spend': 'sum',
                                    'purchases': 'sum',
                                    'roas': 'mean', # Mean ROAS for the group
                                    'cpa': 'mean'  # Mean CPA for the group
                                }).reset_index()

                                demographic_grouped.rename(columns={
                                    'age': 'Age',
                                    'gender': 'Gender',
                                    'spend': 'Amount Spent',
                                    'purchases': 'Purchases',
                                    'roas': 'ROAS',
                                    'cpa': 'CPA'
                                }, inplace=True)                                
                                
                                
                                # ⚠️ Keep numeric for charts
                                demographic_grouped['ROAS'] = demographic_grouped['ROAS'].round(2)

                                # 🧱 Create a copy for table only (to format text safely)
                                if 'ROAS' not in demographic_grouped.columns:
                                    demographic_grouped['ROAS'] = demographic_grouped['Purchases'] / demographic_grouped['Amount Spent'].replace(0, 1)
                                    demographic_grouped['ROAS'] = demographic_grouped['ROAS'].replace([np.inf, -np.inf], 0).fillna(0)

                                # 🧱 Copy for table formatting
                                demographic_table = demographic_grouped.copy()
                                demographic_table['Amount Spent'] = demographic_table['Amount Spent'].apply(lambda x: f"{currency_symbol}{x:,.2f}")
                                #demographic_table['CPA'] = demographic_table['CPA'].apply(lambda x: f"{currency_symbol}{x:,.2f}") NA
                                demographic_table['CPA'] = demographic_table['CPA'].apply(lambda x: f"{currency_symbol}{x:,.2f}" if pd.notna(x) else "N/A")
                                
                                  


                                # 📋 Draw Table
                                #table_data = [demographic_grouped.columns.tolist()] + demographic_grouped.values.tolist()
                                table_data = [demographic_table.columns.tolist()] + demographic_table.values.tolist()
                                
                                # ➤ Grand Total Calculation
                                total_spend = demographic_grouped["Amount Spent"].sum()
                                total_purchases = demographic_grouped["Purchases"].sum()

                                valid_roas = demographic_grouped["ROAS"].dropna()
                                valid_cpa = demographic_grouped["CPA"].dropna()

                                total_roas = valid_roas.mean() if not valid_roas.empty else None
                                total_cpa = valid_cpa.mean() if not valid_cpa.empty else None

                                # ➤ Append Grand Total Row
                                table_data.append([
                                    "Grand Total",                      # Age
                                    "-",                                # Gender (not applicable)
                                    f"{currency_symbol}{total_spend:,.2f}",
                                    int(total_purchases),
                                    f"{total_roas:.2f}" if total_roas is not None else "N/A",
                                    f"{currency_symbol}{total_cpa:,.2f}" if total_cpa is not None else "N/A"
                                ])


                                # Adjust colWidths if needed based on content
                                table_col_widths = [100, 80, 100, 80, 80, 80] # Example widths
                                table = Table(table_data, colWidths=table_col_widths)
                                table.setStyle(TableStyle([
                                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                    ('FONTSIZE', (0, 0), (-1, 0), 10), # Slightly smaller font for table header
                                    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                                    ('FONTNAME', (0, 1), (-1, -1), "DejaVuSans" if currency_symbol == "₹" else "Helvetica"), # Body font
                                    ('FONTSIZE', (0, 1), (-1, -1), 8), # Body font size
                                    ("BACKGROUND", (0, -1), (-1, -1), colors.lightblue),
                                ]))

                                # Calculate table height to position charts below it
                                table_width, table_height = table.wrapOn(c, PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN, PAGE_HEIGHT)
                                table_x = LEFT_MARGIN + (PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN - table_width) / 2 # Center the table
                                table_y_start = PAGE_HEIGHT - TOP_MARGIN - 80 # Position below title
                                table.drawOn(c, table_x, table_y_start - table_height)

                                current_y_pos = table_y_start - table_height - 10 # Start charts 40 units below table

                                # --- Draw Demographic Charts ---
                                # Use demographic_grouped for charts as it's already aggregated
                                
                                # ✅ Rename to expected lowercase for chart functions
                                chart_df = demographic_grouped.rename(columns={"Age": "age", "Gender": "gender"})
                                # 🔍 Debug demographic data before chart generation
                                print("🧪 DEMOGRAPHIC CHART DF COLUMNS:", chart_df.columns.tolist())
                                print("🧪 DEMOGRAPHIC CHART DF HEAD:\n", chart_df.head(2))
                                print("🧪 Amount Spent (sum):", chart_df["Amount Spent"].sum())
                                print("🧪 ROAS values:\n", chart_df["ROAS"].head(2))


                                # Chart layout configs
                                chart_width = 330
                                chart_height = 330
                                chart_padding_x = 50
                                chart_padding_y = 30
                                # Before generating charts, ensure data is properly formatted
                                chart_df = demographic_grouped.rename(columns={
                                    "Amount Spent": "amount_spent",
                                    "Purchases": "purchases",
                                    "ROAS": "roas",
                                    "CPA": "cpa",
                                    "Age": "age",
                                    "Gender": "gender"
                                })

                                # Filter out invalid data
                                chart_df = chart_df[(chart_df['amount_spent'] > 0) & (chart_df['purchases'] >= 0) &(chart_df['roas'] >= 0)]
                                
                                # Calculate starting position with more space
                                current_y_pos = table_y_start - table_height - 100  # Start charts 40 units below table

                                # 🎯 Row 1: Cost + Revenue by Age
                                try:
                                    y_pos = current_y_pos
                                    x_left = LEFT_MARGIN
                                    x_right = PAGE_WIDTH - RIGHT_MARGIN - chart_width

                                    buf = generate_cost_split_by_age_chart(chart_df)
                                    c.drawImage(ImageReader(buf), x_left, y_pos, width=chart_width, height=chart_height, preserveAspectRatio=True)

                                    buf = generate_revenue_split_by_age_chart(chart_df)
                                    c.drawImage(ImageReader(buf), x_right, y_pos, width=chart_width, height=chart_height, preserveAspectRatio=True)

                                    current_y_pos -= (chart_height + chart_padding_y)
                                except Exception as e:
                                    logger.error(f"❌ Row 1 (Cost/Revenue by Age) failed: {e}")
                                    c.setFillColor(colors.red)
                                    c.drawString(LEFT_MARGIN, current_y_pos - 10, "⚠️ Failed to render Cost/Revenue by Age charts")
                                    current_y_pos -= (chart_height + chart_padding_y)

                                # 🎯 Row 2: ROAS by Age + Cost by Gender
                                try:
                                    y_pos = current_y_pos
                                    x_left = LEFT_MARGIN
                                    x_right = PAGE_WIDTH - RIGHT_MARGIN - chart_width

                                    buf = generate_roas_split_by_age_chart(chart_df)
                                    c.drawImage(ImageReader(buf), x_left, y_pos, width=chart_width, height=chart_height, preserveAspectRatio=True)

                                    buf = generate_cost_split_by_gender_chart(chart_df)
                                    c.drawImage(ImageReader(buf), x_right, y_pos, width=chart_width, height=chart_height, preserveAspectRatio=True)

                                    current_y_pos -= (chart_height + chart_padding_y)
                                except Exception as e:
                                    logger.error(f"❌ Row 2 (ROAS by Age + Cost by Gender) failed: {e}")
                                    c.setFillColor(colors.red)
                                    c.drawString(LEFT_MARGIN, current_y_pos - 10, "⚠️ Failed to render ROAS by Age / Cost by Gender charts")
                                    current_y_pos -= (chart_height + chart_padding_y)

                                # 🎯 Row 3: Revenue + ROAS by Gender
                                try:
                                    y_pos = current_y_pos
                                    x_left = LEFT_MARGIN
                                    x_right = PAGE_WIDTH - RIGHT_MARGIN - chart_width

                                    buf = generate_revenue_split_by_gender_chart(chart_df)
                                    c.drawImage(ImageReader(buf), x_left, y_pos, width=chart_width, height=chart_height, preserveAspectRatio=True)

                                    buf = generate_roas_split_by_gender_chart(chart_df)
                                    c.drawImage(ImageReader(buf), x_right, y_pos, width=chart_width, height=chart_height, preserveAspectRatio=True)

                                    current_y_pos -= (chart_height + chart_padding_y)
                                except Exception as e:
                                    logger.error(f"❌ Row 3 (Revenue/ROAS by Gender) failed: {e}")
                                    c.setFillColor(colors.red)
                                    c.drawString(LEFT_MARGIN, current_y_pos - 10, "⚠️ Failed to render Revenue/ROAS by Gender charts")
                                    current_y_pos -= (chart_height + chart_padding_y)

                                # Adjust the page height for the demographic section to ensure all content fits
                                #adjust_page_height(c, {"title": "DEMOGRAPHIC PERFORMANCE", "contains_table": True})

                                                               


                                # 📝 LLM Summary - Dynamic
                                try:
                                    prompt = build_demographic_summary_prompt(demographic_grouped, currency_symbol)
                                    summary_text = run_async_in_thread( generate_llm_content(prompt, demographic_grouped.to_dict()))
    
                                    logger.info("Demographic LLM Summary Generated.")
                                    clean_text = re.sub(r"[*#]", "", summary_text).strip()
                                    clean_text = re.sub(r"\s{2,}", " ", clean_text)

                                    summary_y = current_y_pos - 40

                                    styles = getSampleStyleSheet()
                                    styleN = styles["Normal"]
                                    styleN.fontName = "DejaVuSans" if currency_symbol == "₹" else "Helvetica"
                                    styleN.fontSize = 11
                                    styleN.leading = 14
                                    styleN.textColor = colors.HexColor("#333333")

                                    p = Paragraph(clean_text, styleN)
                                    p_width, p_height = p.wrap(PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN, PAGE_HEIGHT)
                                    p.drawOn(c, LEFT_MARGIN, summary_y - p_height)

                                    draw_footer_cta(c)

                                except Exception as e:
                                    logger.error(f"Demographic LLM Summary generation failed: {e}")
                                    c.setFont("Helvetica", 12)
                                    c.setFillColor(colors.red)
                                    c.drawString(LEFT_MARGIN, current_y_pos - 50, f"⚠️ Unable to generate demographic summary: {str(e)}")
                                    draw_footer_cta(c)
                                #c.showPage()

                        else: # This block executes if demographic_df is not valid for processing
                            logger.warning("Demographic data not available or insufficient for detailed analysis. Skipping section.")
                            c.setFont("Helvetica", 14)
                            c.setFillColor(colors.black)
                            c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT / 2, "⚠️ Demographic data not available for this account or contains no valid entries.")
                            draw_footer_cta(c) # Still draw footer                            
                                                                                                                                                               
                    else:
                        c.showPage()
                        if i < len(sections) - 1:
                            next_section = sections[i + 1]
                            adjust_page_height(c, next_section)

                        # next_section = sections[i + 1]
                        # adjust_page_height(c, next_section)
            #elif section_title.strip().upper() == "DEMOGRAPHIC PERFORMANCE":
                # c.showPage()
                # adjust_page_height(c, {"title": "Demographic Performance", "contains_table": True})
                # #draw_header(c)

                # c.setFont("Helvetica-Bold", 16)
                # c.setFillColor(colors.black)
                # c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - TOP_MARGIN - 30, "Demographic Performance")

                # # ✅ Check for valid demographic data *before* attempting to process it
                # if demographic_df is not None and not demographic_df.empty and \
                #     'age' in demographic_df.columns and 'gender' in demographic_df.columns and \
                #     demographic_df['spend'].sum() > 0: # Ensure there's some spend data too

                #         logger.info("Proceeding with Demographic Performance section as data is valid.")

                #         # Ensure all required columns for aggregation exist and are numeric
                #         for col in ['spend', 'purchase_value', 'purchases']:
                #             if col not in demographic_df.columns:
                #                 demographic_df[col] = 0 # Add missing column with default 0
                #             else:
                #                 demographic_df[col] = pd.to_numeric(demographic_df[col], errors='coerce').fillna(0)

                #                 # Recalculate ROAS and CPA after ensuring numeric columns
                #         demographic_df['roas'] = demographic_df['purchase_value'] / demographic_df['spend'].replace(0, 1)
                #         demographic_df['cpa'] = demographic_df['spend'] / demographic_df['purchases'].replace(0, 1)

                #         demographic_grouped = demographic_df.groupby(['age', 'gender']).agg({
                #             'spend': 'sum',
                #             'purchases': 'sum',
                #             'roas': 'mean', # Mean ROAS for the group
                #             'cpa': 'mean'  # Mean CPA for the group
                #         }).reset_index()

                #         demographic_grouped.rename(columns={
                #             'age': 'Age',
                #             'gender': 'Gender',
                #             'spend': 'Amount Spent',
                #             'purchases': 'Purchases',
                #             'roas': 'ROAS',
                #             'cpa': 'CPA'
                #         }, inplace=True)                                
                                
                                
                #         # ⚠️ Keep numeric for charts
                #         demographic_grouped['ROAS'] = demographic_grouped['ROAS'].round(2)

                #         # 🧱 Create a copy for table only (to format text safely)
                #         if 'ROAS' not in demographic_grouped.columns:
                #             demographic_grouped['ROAS'] = demographic_grouped['Purchases'] / demographic_grouped['Amount Spent'].replace(0, 1)
                #             demographic_grouped['ROAS'] = demographic_grouped['ROAS'].replace([np.inf, -np.inf], 0).fillna(0)

                #         # 🧱 Copy for table formatting
                #         demographic_table = demographic_grouped.copy()
                #         demographic_table['Amount Spent'] = demographic_table['Amount Spent'].apply(lambda x: f"{currency_symbol}{x:,.2f}")
                #         #demographic_table['CPA'] = demographic_table['CPA'].apply(lambda x: f"{currency_symbol}{x:,.2f}") NA
                #         demographic_table['CPA'] = demographic_table['CPA'].apply(lambda x: f"{currency_symbol}{x:,.2f}" if pd.notna(x) else "N/A")
                                
                                  


                #         # 📋 Draw Table
                #         #table_data = [demographic_grouped.columns.tolist()] + demographic_grouped.values.tolist()
                #         table_data = [demographic_table.columns.tolist()] + demographic_table.values.tolist()


                #         # Adjust colWidths if needed based on content
                #         table_col_widths = [100, 80, 100, 80, 80, 80] # Example widths
                #         table = Table(table_data, colWidths=table_col_widths)
                #         table.setStyle(TableStyle([
                #             ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                #             ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                #             ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                #             ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                #             ('FONTSIZE', (0, 0), (-1, 0), 10), # Slightly smaller font for table header
                #             ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                #             ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                #             ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                #             ('FONTNAME', (0, 1), (-1, -1), "DejaVuSans" if currency_symbol == "₹" else "Helvetica"), # Body font
                #             ('FONTSIZE', (0, 1), (-1, -1), 8), # Body font size
                #         ]))

                #         # Calculate table height to position charts below it
                #         table_width, table_height = table.wrapOn(c, PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN, PAGE_HEIGHT)
                #         table_x = LEFT_MARGIN + (PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN - table_width) / 2 # Center the table
                #         table_y_start = PAGE_HEIGHT - TOP_MARGIN - 40 # Position below title
                #         table.drawOn(c, table_x, table_y_start - table_height)

                #         current_y_pos = table_y_start - table_height - 20 # Start charts 40 units below table

                #         # --- Draw Demographic Charts ---
                #         # Use demographic_grouped for charts as it's already aggregated
                                
                #         # ✅ Rename to expected lowercase for chart functions
                #         chart_df = demographic_grouped.rename(columns={"Age": "age", "Gender": "gender"})
                #         # 🔍 Debug demographic data before chart generation
                #         print("🧪 DEMOGRAPHIC CHART DF COLUMNS:", chart_df.columns.tolist())
                #         print("🧪 DEMOGRAPHIC CHART DF HEAD:\n", chart_df.head(2))
                #         print("🧪 Amount Spent (sum):", chart_df["Amount Spent"].sum())
                #         print("🧪 ROAS values:\n", chart_df["ROAS"].head(2))


                #         # Chart layout configs
                #         chart_width = 330
                #         chart_height = 330
                #         chart_padding_x = 50
                #         chart_padding_y = 30
                #         # Before generating charts, ensure data is properly formatted
                #         chart_df = demographic_grouped.rename(columns={
                #             "Amount Spent": "amount_spent",
                #             "Purchases": "purchases",
                #             "ROAS": "roas",
                #             "CPA": "cpa",
                #             "Age": "age",
                #             "Gender": "gender"
                #         })

                #         # Filter out invalid data
                #         chart_df = chart_df[(chart_df['amount_spent'] > 0) & (chart_df['purchases'] >= 0) &(chart_df['roas'] >= 0)]
                                
                #         # Calculate starting position with more space
                #         current_y_pos = table_y_start - table_height - 100  # Start charts 40 units below table

                #         # 🎯 Row 1: Cost + Revenue by Age
                #         try:
                #             y_pos = current_y_pos
                #             x_left = LEFT_MARGIN
                #             x_right = PAGE_WIDTH - RIGHT_MARGIN - chart_width

                #             buf = generate_cost_split_by_age_chart(chart_df)
                #             c.drawImage(ImageReader(buf), x_left, y_pos, width=chart_width, height=chart_height, preserveAspectRatio=True)

                #             buf = generate_revenue_split_by_age_chart(chart_df)
                #             c.drawImage(ImageReader(buf), x_right, y_pos, width=chart_width, height=chart_height, preserveAspectRatio=True)

                #             current_y_pos -= (chart_height + chart_padding_y)
                #         except Exception as e:
                #             logger.error(f"❌ Row 1 (Cost/Revenue by Age) failed: {e}")
                #             c.setFillColor(colors.red)
                #             c.drawString(LEFT_MARGIN, current_y_pos - 10, "⚠️ Failed to render Cost/Revenue by Age charts")
                #             current_y_pos -= (chart_height + chart_padding_y)

                #                 # 🎯 Row 2: ROAS by Age + Cost by Gender
                #         try:
                #             y_pos = current_y_pos
                #             x_left = LEFT_MARGIN
                #             x_right = PAGE_WIDTH - RIGHT_MARGIN - chart_width

                #             buf = generate_roas_split_by_age_chart(chart_df)
                #             c.drawImage(ImageReader(buf), x_left, y_pos, width=chart_width, height=chart_height, preserveAspectRatio=True)

                #             buf = generate_cost_split_by_gender_chart(chart_df)
                #             c.drawImage(ImageReader(buf), x_right, y_pos, width=chart_width, height=chart_height, preserveAspectRatio=True)

                #             current_y_pos -= (chart_height + chart_padding_y)
                #         except Exception as e:
                #             logger.error(f"❌ Row 2 (ROAS by Age + Cost by Gender) failed: {e}")
                #             c.setFillColor(colors.red)
                #             c.drawString(LEFT_MARGIN, current_y_pos - 10, "⚠️ Failed to render ROAS by Age / Cost by Gender charts")
                #             current_y_pos -= (chart_height + chart_padding_y)

                #                 # 🎯 Row 3: Revenue + ROAS by Gender
                #         try:
                #             y_pos = current_y_pos
                #             x_left = LEFT_MARGIN
                #             x_right = PAGE_WIDTH - RIGHT_MARGIN - chart_width

                #             buf = generate_revenue_split_by_gender_chart(chart_df)
                #             c.drawImage(ImageReader(buf), x_left, y_pos, width=chart_width, height=chart_height, preserveAspectRatio=True)

                #             buf = generate_roas_split_by_gender_chart(chart_df)
                #             c.drawImage(ImageReader(buf), x_right, y_pos, width=chart_width, height=chart_height, preserveAspectRatio=True)

                #             current_y_pos -= (chart_height + chart_padding_y)
                #         except Exception as e:
                #             logger.error(f"❌ Row 3 (Revenue/ROAS by Gender) failed: {e}")
                #             c.setFillColor(colors.red)
                #             c.drawString(LEFT_MARGIN, current_y_pos - 10, "⚠️ Failed to render Revenue/ROAS by Gender charts")
                #             current_y_pos -= (chart_height + chart_padding_y)

                #                 # Adjust the page height for the demographic section to ensure all content fits
                #                 #adjust_page_height(c, {"title": "DEMOGRAPHIC PERFORMANCE", "contains_table": True})

                #         # 📝 LLM Summary - Dynamic
                #         try:
                #             prompt = build_demographic_summary_prompt(demographic_grouped, currency_symbol)
                #             summary_text = run_async_in_thread( generate_llm_content(prompt, demographic_grouped.to_dict()))
    
                #             logger.info("Demographic LLM Summary Generated.")
                #             clean_text = re.sub(r"[*#]", "", summary_text).strip()
                #             clean_text = re.sub(r"\s{2,}", " ", clean_text)

                #             summary_y = current_y_pos - 40

                #             styles = getSampleStyleSheet()
                #             styleN = styles["Normal"]
                #             styleN.fontName = "DejaVuSans" if currency_symbol == "₹" else "Helvetica"
                #             styleN.fontSize = 11
                #             styleN.leading = 14
                #             styleN.textColor = colors.HexColor("#333333")

                #             p = Paragraph(clean_text, styleN)
                #             p_width, p_height = p.wrap(PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN, PAGE_HEIGHT)
                #             p.drawOn(c, LEFT_MARGIN, summary_y - p_height)

                #             draw_footer_cta(c)

                #         except Exception as e:
                #             logger.error(f"Demographic LLM Summary generation failed: {e}")
                #             c.setFont("Helvetica", 12)
                #             c.setFillColor(colors.red)
                #             c.drawString(LEFT_MARGIN, current_y_pos - 50, f"⚠️ Unable to generate demographic summary: {str(e)}")
                #             draw_footer_cta(c)
                #                 #c.showPage()

                # else: # This block executes if demographic_df is not valid for processing
                #     logger.warning("Demographic data not available or insufficient for detailed analysis. Skipping section.")
                #     c.setFont("Helvetica", 14)
                #     c.setFillColor(colors.black)
                #     c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT / 2, "⚠️ Demographic data not available for this account or contains no valid entries.")
                #     draw_footer_cta(c) # Still draw footer 
            # elif section_title.strip().upper() == "DEMOGRAPHIC PERFORMANCE":
            #     continue  

                   
            elif section_title.strip().upper() == "PLATFORM LEVEL PERFORMANCE":
                c.showPage()
                # Ensure platform_df is valid before processing
                if platform_df is not None and not platform_df.empty:
                    
                    adjust_page_height(c, section)
                    draw_header(c)
                    from services.chart_utils import (
                        generate_platform_split_charts,
                        generate_platform_roas_chart,
                        generate_platform_cost_line_chart,
                        generate_platform_revenue_line_chart,
                    )
                    from services.deepseek_audit import generate_platform_summary, group_by_platform # Import group_by_platform here

                    

                    c.setFont("Helvetica-Bold", 20)
                    c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT - TOP_MARGIN - 30, "Platform Level Performance")

                    # Process the platform_df to get summarized data
                    platform_summary_df = group_by_platform(platform_df, currency_symbol)
                    
                    
                    # Add Grand Total row for table
                    # ➤ Grand Total Calculation (safe handling of NaNs)
                    total_spend = platform_summary_df['spend'].sum()
                    total_revenue = platform_summary_df['purchase_value'].sum()
                    total_purchases = platform_summary_df['purchases'].sum()

                    valid_roas = platform_summary_df['roas'].dropna()
                    valid_cpa = platform_summary_df['cpa'].dropna()

                    total_roas = valid_roas.mean() if not valid_roas.empty else None
                    total_cpa = valid_cpa.mean() if not valid_cpa.empty else None

                    # ➤ Append Grand Total as a dict row
                    total_row = {
                        'platform': 'Grand Total',
                        'spend': total_spend,
                        'purchase_value': total_revenue,
                        'purchases': total_purchases,
                        'roas': total_roas,
                        'cpa': total_cpa
                    }

                    platform_table_data = pd.concat([platform_summary_df, pd.DataFrame([total_row])], ignore_index=True)



                    # Format table data
                    table_data = [["Platform", "Amount Spent", "Revenue", "Purchases", "ROAS", "CPA"]]
                    for _, row in platform_table_data.iterrows():
                        table_data.append([
                            row['platform'],
                            f"{currency_symbol}{row['spend']:,.2f}" if pd.notna(row['spend']) else "-",
                            f"{currency_symbol}{row['purchase_value']:,.2f}" if pd.notna(row['purchase_value']) else "-",
                            int(row['purchases']) if pd.notna(row['purchases']) else "-",
                            f"{row['roas']:.2f}" if pd.notna(row['roas']) else "-",
                            #f"{currency_symbol}{row['cpa']:.2f}" if pd.notna(row['cpa']) else "-" NA
                            f"{currency_symbol}{row['cpa']:.2f}" if pd.notna(row['cpa']) else "N/A"
                        ])

                    performance_table = Table(table_data, repeatRows=1, colWidths=[200, 120, 120, 100, 90, 90])
                    performance_table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                        ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans" if currency_symbol == "₹" else "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("BACKGROUND", (0, -1), (-1, -1), colors.lightblue),
                    ]))
                    table_y = PAGE_HEIGHT - 300
                    performance_table.wrapOn(c, PAGE_WIDTH, PAGE_HEIGHT)
                    performance_table.drawOn(c, LEFT_MARGIN, table_y)

                    # Charts (2 pie + 3 line/bar)
                    split_charts = generate_platform_split_charts(platform_df)
                    roas_chart_img_buf = generate_platform_roas_chart(platform_df)
                    cost_line_chart_img_buf = generate_platform_cost_line_chart(platform_df)
                    revenue_line_chart_img_buf = generate_platform_revenue_line_chart(platform_df)

                    
                    chart_y = table_y - 420
                    if len(split_charts) >= 2:
                        donut_width = 300
                        donut_height = 300
                        # Cost Split (left)
                        img1 = ImageReader(split_charts[0][1])
                        c.drawImage(img1, LEFT_MARGIN, chart_y, width=donut_width, height=donut_height)


                        # Revenue Split (right)
                        img2 = ImageReader(split_charts[1][1])
                        c.drawImage(img2, PAGE_WIDTH - RIGHT_MARGIN - donut_width, chart_y, width=donut_width, height=donut_height)


                    # ROAS Bar Chart
                    #roas_chart_img_buf = generate_platform_roas_chart(platform_df) # Pass original platform_df
                    roas_y = chart_y - 360
                    img3 = ImageReader(roas_chart_img_buf)
                    c.drawImage(img3, LEFT_MARGIN, roas_y, width=PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN, height=300)

                    # Cost Line Chart
                    #cost_line_chart_img_buf = generate_platform_cost_line_chart(platform_df) # Pass original platform_df
                    cost_y = roas_y - 350
                    img4 = ImageReader(cost_line_chart_img_buf)
                    c.drawImage(img4, LEFT_MARGIN, cost_y, width=PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN, height=300)

                    # Revenue Line Chart
                    #revenue_line_chart_img_buf = generate_platform_revenue_line_chart(platform_df) # Pass original platform_df
                    rev_y = cost_y - 350
                    img5 = ImageReader(revenue_line_chart_img_buf)
                    c.drawImage(img5, LEFT_MARGIN, rev_y, width=PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN, height=300)

                    # Add LLM summary
                    try:
                        summary_text = run_async_in_thread(generate_platform_summary(platform_summary_df, currency_symbol)) # Pass summarized data to LLM
                        c.setFont("Helvetica", 11)
                        lines = simpleSplit(summary_text, "Helvetica", 11, PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN)
                        text_y = rev_y - 40
                        for line in lines:
                            c.drawString(LEFT_MARGIN, text_y, line)
                            text_y -= 14
                    except Exception as e:
                        print("⚠️ Failed to generate platform summary:", str(e))
                else:
                    c.setFont("Helvetica", 14)
                    c.setFillColor(colors.black)
                    c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT / 2, "⚠️ Platform data not available for this account.")
                    draw_footer_cta(c)
                

            
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
                    # c.setStrokeColor(colors.HexColor("#007bff"))
                    # c.setLineWidth(8)
                    # c.line(text_x - 10, BOTTOM_MARGIN, text_x - 10, PAGE_HEIGHT - TOP_MARGIN)

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