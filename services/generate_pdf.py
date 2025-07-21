from reportlab.lib.pagesizes import landscape
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.utils import simpleSplit
import io
import os
import asyncio
import threading
from fastapi.responses import StreamingResponse
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.lib.colors import HexColor # Ensure HexColor is imported if used
import re
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd
import numpy as np # Make sure numpy is imported for np.all and np.isfinite

# Assuming these are correctly imported from services.chart_utils
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
    generate_roas_split_by_gender_chart,
    generate_frequency_over_time_chart,
    generate_cpm_over_time_chart
)

from services.deepseek_audit import (
    generate_llm_content,
    build_demographic_summary_prompt,
    generate_adset_summary,
    generate_ad_summary,
    generate_roas_summary_text,
    generate_ad_fatigue_summary # Make sure this is also imported if used
)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Re-declarations of functions and global variables (as per your original file structure) ---
# It's crucial that these match your actual generate_pdf.py file.
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

font_path = os.path.join("assets", "fonts", "DejaVuSans.ttf")
pdfmetrics.registerFont(TTFont("DejaVuSans", font_path))

PAGE_WIDTH = 1000
PAGE_HEIGHT = 700 # Default, will be adjusted
LEFT_MARGIN = inch
RIGHT_MARGIN = inch
TOP_MARGIN = 1.2 * inch
BOTTOM_MARGIN = inch

LOGO_WIDTH = 240
LOGO_HEIGHT = 45
LOGO_Y_OFFSET = PAGE_HEIGHT - TOP_MARGIN + 10 # Initial value, adjusted by adjust_page_height

LOGO_PATH = os.path.join(BASE_DIR, "..", "assets", "Data_Vinci_Logo.png")

def set_font_with_currency(c, currency_symbol, fallback_font="Helvetica", size=12):
    if currency_symbol == "₹":
        c.setFont("DejaVuSans", size)
    else:
        c.setFont(fallback_font, size)

def adjust_page_height(c, section: dict):
    global PAGE_HEIGHT, LOGO_Y_OFFSET, TOP_MARGIN
    title = section.get("title", "").upper().strip()
    if "CAMPAIGN PERFORMANCE OVERVIEW" in title:
        PAGE_HEIGHT = 800
    elif title == "CAMPAIGN PERFORMANCE SUMMARY":
        PAGE_HEIGHT = 2300
    elif title == "3 CHARTS SECTION":
        PAGE_HEIGHT = 1400
    elif title == "ADSET LEVEL PERFORMANCE":
        PAGE_HEIGHT = 2500
    elif title == "AD LEVEL PERFORMANCE":
        PAGE_HEIGHT = 4000
    elif title == "AD FATIGUE ANALYSIS":
        PAGE_HEIGHT = 4500
    elif title == "DEMOGRAPHIC PERFORMANCE":
        PAGE_HEIGHT = 3000
    else:
        PAGE_HEIGHT = 700 # Default for other text-based sections

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

        value_cleaned = str(value).replace("■", "").strip()

        col = i % cols
        row = i // cols
        x = x_start + col * (card_width + padding_x)
        y_offset = row * (card_height + padding_y)
        card_y = y - y_offset

        c.setFillColor(colors.HexColor("#e1fbd2"))  # soft green background
        c.roundRect(x, card_y - card_height, card_width, card_height, 10, fill=1, stroke=0)

        c.setFillColor(colors.HexColor("#222222"))
        c.setFont("Helvetica-Bold", 12)
        c.drawCentredString(x + card_width / 2, card_y - 18, label.strip())
        
        set_font_with_currency(c, value_cleaned.strip()[0], size=12)
        c.drawCentredString(x + card_width / 2, card_y - 38, value_cleaned)

# --- Start of the generate_pdf_report function's relevant section ---
def generate_pdf_report(sections: list, ad_insights_df=None, full_ad_insights_df=None, currency_symbol=None, split_charts=None, demographic_df=None) -> StreamingResponse:
    global PAGE_HEIGHT, LOGO_Y_OFFSET, TOP_MARGIN

    if currency_symbol is None:
        currency_symbol = "₹"
        
    if ad_insights_df is not None:
        if 'reach' not in ad_insights_df.columns and 'impressions' in ad_insights_df.columns:
            logger.warning("⚠️ 'reach' missing in ad_insights_df, using impressions as fallback")
            ad_insights_df['reach'] = ad_insights_df['impressions']
        elif 'reach' not in ad_insights_df.columns:
            logger.warning("⚠️ Both 'reach' and 'impressions' missing - setting reach to 1 to avoid division errors")
            ad_insights_df['reach'] = 1

    if full_ad_insights_df is not None:
        if 'reach' not in full_ad_insights_df.columns and 'impressions' in full_ad_insights_df.columns:
            logger.warning("⚠️ 'reach' missing in full_ad_insights_df, using impressions as fallback")
            full_ad_insights_df['reach'] = full_ad_insights_df['impressions']
        elif 'reach' not in full_ad_insights_df.columns:
            logger.warning("⚠️ Both 'reach' and 'impressions' missing - setting reach to 1 to avoid division errors")
            full_ad_insights_df['reach'] = 1
        
    if ad_insights_df is not None and 'roas' not in ad_insights_df.columns:
        logger.warning("⚠️ 'roas' missing in ad_insights_df, calculating fallback.")
        ad_insights_df['roas'] = ad_insights_df['purchase_value'] / ad_insights_df['spend'].replace(0, 1)
        ad_insights_df['roas'] = ad_insights_df['roas'].fillna(0).round(2)

    if full_ad_insights_df is not None and 'roas' not in full_ad_insights_df.columns:
        logger.warning("⚠️ 'roas' missing in full_ad_insights_df, calculating fallback.")
        full_ad_insights_df['roas'] = full_ad_insights_df['purchase_value'] / full_ad_insights_df['spend'].replace(0, 1)
        full_ad_insights_df['roas'] = full_ad_insights_df['roas'].fillna(0).round(2)

    try:
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer)   

        for i, section in enumerate(sections):  
            # --- START FIX FOR PAGE BREAKS AND LAYOUT ---
            # This logic ensures each major section starts on a new page.
            # Only the first section doesn't get a c.showPage() before it.
            if i > 0:
                c.showPage()
            
            adjust_page_height(c, section)
            section_title = section.get("title", "Untitled Section").strip().upper()
            content = section.get("content", "No content available.")
            charts = section.get("charts", [])
            draw_header(c)

            current_y = PAGE_HEIGHT - TOP_MARGIN - 30 # Initialize current_y for section content

            if section_title == "KEY METRICS":
                # ( ... Your existing KEY METRICS page content - untouched ... )
                # Page 1: Key Metrics Header & Cards
                c.setFont("Helvetica-Bold", 24)
                c.setFillColor(colors.black)
                c.drawCentredString(PAGE_WIDTH / 2, current_y, "Key Metrics")
                current_y -= 40 # Adjust for title spacing

                metric_lines = [line for line in content.split("\n") if ":" in line and "Last 30" not in line]
                metrics = dict(line.split(":", 1) for line in metric_lines)
                draw_metrics_grid(c, metrics, current_y) 
                current_y -= 180 # Adjust for grid height

                # Page 2: Trend Heading & Paragraph
                c.showPage()
                adjust_page_height(c, section) # Re-adjust for current section if needed
                draw_header(c)
                current_y = PAGE_HEIGHT - TOP_MARGIN - 30 # Reset Y for new page

                c.setFont("Helvetica-Bold", 20)
                c.drawString(LEFT_MARGIN, current_y, "Last 30 Days Trend Section")
                current_y -= 30

                paragraph = (
                    "The following section presents daily trend of the Key Metrics Identified in the previous section. "
                    "This helps the business analyse the daily variance in the business KPIs and also helps in correlating "
                    "how one metric affects the others."
                )
                set_font_with_currency(c, currency_symbol, size=12)
                text_width = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
                lines = simpleSplit(paragraph, "DejaVuSans" if currency_symbol == "₹" else "Helvetica", 12, text_width)
                for line in lines:
                    c.drawString(LEFT_MARGIN, current_y, line)
                    current_y -= 14
                current_y -= 20 # Add space after paragraph

                # Chart 1 — Amount Spent vs Purchase Conversion Value
                if charts:
                    try:
                        chart_title = charts[0][0]
                        c.setFont("Helvetica-Bold", 16)
                        c.drawCentredString(PAGE_WIDTH / 2, current_y, chart_title)
                        current_y -= 20

                        chart_width = PAGE_WIDTH - 1.5 * LEFT_MARGIN
                        chart_height = 350
                        chart_x = (PAGE_WIDTH - chart_width) / 2

                        img1 = ImageReader(charts[0][1])
                        c.drawImage(img1, chart_x, current_y - chart_height, width=chart_width, height=chart_height, preserveAspectRatio=True)
                        current_y -= (chart_height + 40)
                    except Exception as e:
                        logger.warning(f"⚠️ Chart 1 render error: {str(e)}") 
                
                # Render charts 2-4 on a new page
                c.showPage()
                adjust_page_height(c, {"title": "3 Charts Section", "contains_table": False})
                draw_header(c)
                current_y = PAGE_HEIGHT - TOP_MARGIN - 30

                chart_titles = [
                    "Purchases vs ROAS",
                    "CPA vs Link CPC",
                    "Click to Conversion vs CTR"
                ]

                chart_spacing = 70
                
                for idx, (title, chart_buf) in enumerate(charts[1:4]):
                    try:
                        c.setFont("Helvetica-Bold", 14)
                        c.drawCentredString(PAGE_WIDTH / 2, current_y, title)
                        current_y -= 20

                        img = ImageReader(chart_buf)
                        c.drawImage(
                            img,
                            (PAGE_WIDTH - chart_width) / 2,
                            current_y - chart_height,
                            width=chart_width,
                            height=chart_height,
                            preserveAspectRatio=True
                        )
                        current_y -= (chart_height + chart_spacing)
                    except Exception as e:
                        logger.warning(f"⚠️ Error rendering chart {title}: {str(e)}")
                                
                draw_footer_cta(c)
 
            elif section_title == "CAMPAIGN PERFORMANCE OVERVIEW":
                # ( ... Your existing Campaign Performance Overview content - untouched ... )
                adjust_page_height(c, {"title": "Campaign Performance Overview", "contains_table": True})
                draw_header(c)
                current_y = PAGE_HEIGHT - TOP_MARGIN - 30

                c.setFont("Helvetica-Bold", 18)
                c.setFillColor(colors.black)
                c.drawCentredString(PAGE_WIDTH / 2, current_y, "Campaign Performance Overview")
                current_y -= 40

                if ad_insights_df is not None and not ad_insights_df.empty:
                    ad_insights_df = ad_insights_df.sort_values('date')
                    
                    totals = {
                        'spend': ad_insights_df['spend'].sum(),
                        'purchases': ad_insights_df['purchases'].sum(),
                        'purchase_value': ad_insights_df['purchase_value'].sum(),
                        'cpa': ad_insights_df['cpa'].mean(),
                        'impressions': ad_insights_df['impressions'].sum(),
                        'ctr': ad_insights_df['ctr'].mean(),
                        'clicks': ad_insights_df['clicks'].sum(),
                        'click_to_conversion': ad_insights_df['click_to_conversion'].mean(),
                        'roas': ad_insights_df['roas'].mean()
                    }
                    table_data = [["Day", "Amount spent", "Purchases", "Purchases conversion value", "CPA", "Impressions","CTR", "Link clicks", "Click To Conversion", "ROAS"]]

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
                            f"{row['roas']:.2f}",
                        ])

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

                    summary_table = Table(table_data, repeatRows=1, colWidths=[90]*10)
                    summary_table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                        ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans" if currency_symbol == "₹" else "Helvetica"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("BACKGROUND", (0, -1), (-1, -1), colors.lightblue),
                    ]))

                    table_y_pos = current_y - summary_table._calcHeight(summary_table._argW) - 20
                    summary_table.wrapOn(c, PAGE_WIDTH, PAGE_HEIGHT)
                    summary_table.drawOn(c, LEFT_MARGIN, table_y_pos)
                    current_y = table_y_pos - 20
                    
                else:
                    c.setFont("Helvetica", 12)
                    c.drawCentredString(PAGE_WIDTH / 2, current_y - 50, "No daily performance data available.")
                    current_y -= 100

                draw_footer_cta(c)

            elif section_title == "CAMPAIGN PERFORMANCE SUMMARY":
                # ( ... Your existing Campaign Level Performance Summary content - untouched ... )
                adjust_page_height(c, section)
                draw_header(c)
                current_y = PAGE_HEIGHT - TOP_MARGIN - 30

                c.setFont("Helvetica-Bold", 16)
                c.drawCentredString(PAGE_WIDTH / 2, current_y, "Campaign Level Performance")
                current_y -= 40

                df = full_ad_insights_df.copy()
                df = df[df['campaign_name'].notna()]
                if not df.empty:
                    numeric_cols = ['spend', 'purchase_value', 'purchases']
                    for col in numeric_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

                    grouped_campaigns = df.groupby('campaign_name').agg({
                        'spend': 'sum',
                        'purchase_value': 'sum',
                        'purchases': 'sum'
                    }).reset_index()

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

                    performance_table = Table(table_data, repeatRows=1, colWidths=[260, 140, 140, 100, 100, 100])
                    performance_table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                        ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans" if currency_symbol == "₹" else "Helvetica"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("BACKGROUND", (0, -1), (-1, -1), colors.lightblue),
                    ]))

                    table_height_actual = performance_table._calcHeight(performance_table._argW)
                    table_y_pos = current_y - table_height_actual - 20
                    performance_table.wrapOn(c, PAGE_WIDTH, PAGE_HEIGHT)
                    performance_table.drawOn(c, LEFT_MARGIN, table_y_pos)
                    current_y = table_y_pos - 40

                else:
                    c.setFont("Helvetica", 12)
                    c.drawCentredString(PAGE_WIDTH / 2, current_y - 50, "No campaign data available for detailed analysis.")
                    current_y -= 100
                
                if split_charts and len(split_charts) >= 3:
                    donut_width = 410
                    donut_height = 410
                    padding_y = 40
                    
                    cost_x = LEFT_MARGIN
                    c.setStrokeColor(colors.lightgrey)
                    c.setLineWidth(1)
                    c.roundRect(cost_x, current_y - donut_height, donut_width, donut_height, radius=8, fill=0, stroke=1)
                    img1 = ImageReader(split_charts[0][1])
                    c.drawImage(img1, cost_x, current_y - donut_height, width=donut_width, height=donut_height)

                    revenue_x = PAGE_WIDTH - RIGHT_MARGIN - donut_width
                    c.setStrokeColor(colors.lightgrey)
                    c.setLineWidth(1)
                    c.roundRect(revenue_x, current_y - donut_height, donut_width, donut_height, radius=8, fill=0, stroke=1)
                    img2 = ImageReader(split_charts[1][1])
                    c.drawImage(img2, revenue_x, current_y - donut_height, width=donut_width, height=donut_height)

                    current_y -= (donut_height + padding_y)

                    roas_width = 740
                    roas_height = 320
                    roas_x = (PAGE_WIDTH - roas_width) / 2
                    c.setStrokeColor(colors.lightgrey)
                    c.setLineWidth(1)
                    c.roundRect(roas_x, current_y - roas_height, roas_width, roas_height, radius=8, fill=0, stroke=1)
                    img3 = ImageReader(split_charts[2][1])
                    c.drawImage(img3, roas_x, current_y - roas_height, width=roas_width, height=roas_height)
                    current_y -= (roas_height + 60)

                try:
                    cost_by_campaign_chart = generate_cost_by_campaign_chart(full_ad_insights_df)
                    c.setFont("Helvetica-Bold", 16)
                    c.drawCentredString(PAGE_WIDTH / 2, current_y, "Cost by Campaigns")
                    current_y -= 20
                    
                    chart_width = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
                    chart_height = 420
                    c.drawImage(ImageReader(cost_by_campaign_chart[1]), LEFT_MARGIN, current_y - chart_height, width=chart_width, height=chart_height, preserveAspectRatio=True)
                    current_y -= (chart_height + 40)
                except Exception as e:
                    logger.warning(f"⚠️ Error rendering Cost by Campaigns chart: {str(e)}")
                    current_y -= 50

                try:
                    rev_chart = generate_revenue_by_campaign_chart(full_ad_insights_df)
                    c.setFont("Helvetica-Bold", 16)
                    c.drawCentredString(PAGE_WIDTH / 2, current_y, "Revenue by Campaigns")
                    current_y -= 20
                    
                    chart_width = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
                    chart_height = 420
                    c.drawImage(ImageReader(rev_chart[1]), LEFT_MARGIN, current_y - chart_height, width=chart_width, height=chart_height, preserveAspectRatio=True)
                    current_y -= (chart_height + 40)
                except Exception as e:
                    logger.warning(f"⚠️ Error rendering Revenue by Campaigns chart: {str(e)}")
                    current_y -= 50

                try:
                    summary_text = run_async_in_thread(generate_roas_summary_text(full_ad_insights_df, currency_symbol))
                    logger.info("📄 Campaign LLM Summary Generated")
                    clean_text = re.sub(r"[*#]", "", summary_text).strip()
                    clean_text = re.sub(r"\s{2,}", " ", clean_text)

                    styles = getSampleStyleSheet()
                    styleN = styles["Normal"]
                    styleN.fontName = "DejaVuSans" if currency_symbol == "₹" else "Helvetica"
                    styleN.fontSize = 11
                    styleN.leading = 14
                    styleN.textColor = colors.HexColor("#333333")

                    p = Paragraph(clean_text, styleN)
                    p_width, p_height = p.wrap(PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN, PAGE_HEIGHT)
                    
                    current_y -= (p_height + 20)
                    p.drawOn(c, LEFT_MARGIN, current_y)

                except Exception as e:
                    logger.warning(f"⚠️ Campaign LLM Summary generation failed: {str(e)}")
                    c.setFont("Helvetica", 12)
                    c.setFillColor(colors.red)
                    c.drawString(LEFT_MARGIN, current_y - 50, f"⚠️ Unable to generate campaign summary: {str(e)}")
                    current_y -= 100
                    
                draw_footer_cta(c)

            elif section_title == "ADSET LEVEL PERFORMANCE":
                # ( ... Your existing Adset Level Performance content - untouched ... )
                adjust_page_height(c, section)
                draw_header(c)
                current_y = PAGE_HEIGHT - TOP_MARGIN - 30

                c.setFont("Helvetica-Bold", 16)
                c.setFillColor(colors.black)
                c.drawCentredString(PAGE_WIDTH / 2, current_y, "Adset Level Performance")
                current_y -= 40

                df = full_ad_insights_df.copy()
                df = df[df['adset_name'].notna()]
                if not df.empty:
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
                            f"{row['roas']:.2f}",
                            f"{currency_symbol}{row['cpa']:.2f}"
                        ])

                    summary_table = Table(table_data, repeatRows=1, colWidths=[270, 130, 130, 90, 90, 110])
                    summary_table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                        ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans" if currency_symbol == "₹" else "Helvetica"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("BACKGROUND", (0, -1), (-1, -1), colors.lightblue),
                    ]))
                    table_height_actual = summary_table._calcHeight(summary_table._argW)
                    table_y_pos = current_y - table_height_actual - 20
                    summary_table.wrapOn(c, PAGE_WIDTH, PAGE_HEIGHT)
                    summary_table.drawOn(c, LEFT_MARGIN, table_y_pos)
                    current_y = table_y_pos - 40

                    donut_width = 380
                    donut_height = 380
                    donut_padding_y = 40

                    cost_x = LEFT_MARGIN
                    c.setStrokeColor(colors.lightgrey)
                    c.setLineWidth(1)
                    c.roundRect(cost_x, current_y - donut_height, donut_width, donut_height, radius=8, fill=0, stroke=1)
                    try:
                        fig1 = draw_donut_chart(top_spend.values, top_spend.index, "Cost Split")
                        img1 = ImageReader(generate_chart_image(fig1))
                        c.drawImage(img1, cost_x, current_y - donut_height, width=donut_width, height=donut_height)
                    except Exception as e:
                        c.setFont("Helvetica", 10)
                        c.setFillColor(colors.red)
                        c.drawString(cost_x + 20, current_y - donut_height + donut_height / 2, f"⚠️ Cost Split chart failed: {str(e)}")

                    revenue_x = PAGE_WIDTH - RIGHT_MARGIN - donut_width
                    c.setStrokeColor(colors.lightgrey)
                    c.setLineWidth(1)
                    c.roundRect(revenue_x, current_y - donut_height, donut_width, donut_height, radius=8, fill=0, stroke=1)
                    try:
                        fig2 = draw_donut_chart(top_revenue.values, top_revenue.index, "Revenue Split")
                        img2 = ImageReader(generate_chart_image(fig2))
                        c.drawImage(img2, revenue_x, current_y - donut_height, width=donut_width, height=donut_height)
                    except Exception as e:
                        logger.warning(f"⚠️ Error rendering Revenue Split: {str(e)}")
                    
                    current_y -= (donut_height + donut_padding_y)

                    roas_width = 770
                    roas_height = 280
                    roas_x = (PAGE_WIDTH - roas_width) / 2
                    c.setStrokeColor(colors.lightgrey)
                    c.setLineWidth(1)
                    c.roundRect(roas_x, current_y - roas_height, roas_width, roas_height, radius=8, fill=0, stroke=1)
                    try:
                        fig3 = draw_roas_split_bar_chart(top_roas)
                        img3 = ImageReader(generate_chart_image(fig3))
                        c.drawImage(img3, roas_x, current_y - roas_height, width=roas_width, height=roas_height)
                    except Exception as e:
                        logger.warning(f"⚠️ Error rendering ROAS Split: {str(e)}")
                    current_y -= (roas_height + 60)

                    try:
                        cost_chart = generate_cost_by_adset_chart(full_ad_insights_df)
                        c.setFont("Helvetica-Bold", 14)
                        c.setFillColor(colors.black)
                        c.drawCentredString(PAGE_WIDTH / 2, current_y, "Cost by Adsets")
                        current_y -= 20
                        
                        chart_width_line = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
                        chart_height_line = 480
                        c.drawImage(ImageReader(cost_chart[1]), LEFT_MARGIN, current_y - chart_height_line, width=chart_width_line, height=chart_height_line)
                        current_y -= (chart_height_line + 60)
                    except Exception as e:
                        logger.warning(f"⚠️ Error rendering Cost by Adsets chart: {str(e)}")
                        current_y -= 50

                    try:
                        revenue_chart = generate_revenue_by_adset_chart(full_ad_insights_df)
                        c.setFont("Helvetica-Bold", 14)
                        c.drawCentredString(PAGE_WIDTH / 2, current_y, "Revenue by Adsets")
                        current_y -= 20
                        
                        c.drawImage(ImageReader(revenue_chart[1]), LEFT_MARGIN, current_y - chart_height_line, width=chart_width_line, height=chart_height_line)
                        current_y -= (chart_height_line + 60)
                    except Exception as e:
                        logger.warning(f"⚠️ Error rendering Revenue by Adsets chart: {str(e)}")
                        current_y -= 50

                    try:                              
                        summary_text = run_async_in_thread(generate_adset_summary(full_ad_insights_df, currency_symbol))
                        logger.info("📄 Adset LLM Summary Generated")

                        clean_text = re.sub(r"[*#]", "", summary_text).strip()
                        clean_text = re.sub(r"\s{2,}", " ", clean_text)
                        
                        styles = getSampleStyleSheet()
                        styleN = styles["Normal"]
                        styleN.fontName = "DejaVuSans" if currency_symbol == "₹" else "Helvetica"
                        styleN.fontSize = 11
                        styleN.leading = 14
                        styleN.textColor = colors.HexColor("#333333")

                        p = Paragraph(clean_text, styleN)
                        p_width, p_height = p.wrap(PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN, PAGE_HEIGHT)
                        
                        current_y -= (p_height + 20)
                        p.drawOn(c, LEFT_MARGIN, current_y)
                                
                    except Exception as e:
                        logger.warning(f"⚠️ Adset LLM Summary generation failed: {str(e)}")  
                        c.setFont("Helvetica", 12)
                        c.setFillColor(colors.red)
                        c.drawString(LEFT_MARGIN, current_y - 50, f"⚠️ Unable to generate adset summary: {str(e)}")
                        current_y -= 100
                                
                else:
                    c.setFont("Helvetica", 12)
                    c.drawCentredString(PAGE_WIDTH / 2, current_y - 50, "No adset data available for detailed analysis.")
                    current_y -= 100

                draw_footer_cta(c)

            elif section_title == "AD LEVEL PERFORMANCE":
                # ( ... Your existing Ad Level Performance content - untouched ... )
                adjust_page_height(c, section)
                draw_header(c)
                current_y = PAGE_HEIGHT - TOP_MARGIN - 30

                c.setFont("Helvetica-Bold", 14)
                c.drawCentredString(PAGE_WIDTH / 2, current_y, "Ad Level Performance")
                current_y -= 40

                ad_df = full_ad_insights_df.copy()
                ad_df = ad_df[ad_df['ad_name'].notna()]
                if not ad_df.empty:
                    ad_df['spend'] = pd.to_numeric(ad_df['spend'], errors='coerce').fillna(0)
                    ad_df['purchase_value'] = pd.to_numeric(ad_df['purchase_value'], errors='coerce').fillna(0)
                    ad_df['purchases'] = pd.to_numeric(ad_df['purchases'], errors='coerce').fillna(0)
                    ad_df['roas'] = ad_df['purchase_value'] / ad_df['spend'].replace(0, 1)
                    ad_df['cpa'] = ad_df['spend'] / ad_df['purchases'].replace(0, 1)

                    ad_grouped = ad_df.groupby('ad_name').agg({
                        'spend': 'sum',
                        'purchase_value': 'sum',
                        'purchases': 'sum',
                        'roas': 'mean',
                        'cpa': 'mean'
                    }).reset_index()

                    ad_table_data = [["Ad Name", "Amount Spent", "Revenue", "Purchases", "ROAS", "CPA"]]
                    for _, row in ad_grouped.iterrows():
                        ad_table_data.append([
                            row['ad_name'],
                            f"{currency_symbol}{row['spend']:.2f}",
                            f"{currency_symbol}{row['purchase_value']:.2f}",
                            int(row['purchases']),
                            f"{row['roas']:.2f}",
                            f"{currency_symbol}{row['cpa']:.2f}"
                        ])

                    ad_summary_table = Table(ad_table_data, repeatRows=1, colWidths=[250, 130, 130, 80, 90, 120])
                    ad_summary_table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                        ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans" if currency_symbol == "₹" else "Helvetica"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("BACKGROUND", (0, -1), (-1, -1), colors.lightblue),
                    ]))
                    table_height_actual = ad_summary_table._calcHeight(ad_summary_table._argW)
                    table_y_pos = current_y - table_height_actual - 20
                    ad_summary_table.wrapOn(c, PAGE_WIDTH, PAGE_HEIGHT)
                    ad_summary_table.drawOn(c, LEFT_MARGIN, table_y_pos)
                    current_y = table_y_pos - 40

                    donut_width, donut_height = 400, 400
                    padding_inner = 20

                    cost_x = LEFT_MARGIN
                    c.setStrokeColor(colors.lightgrey)
                    c.setLineWidth(1)
                    c.roundRect(cost_x, current_y - donut_height, donut_width, donut_height, radius=8, fill=0, stroke=1)
                    fig1 = draw_donut_chart(top_ad_spend.values, top_ad_spend.index, "")
                    c.drawImage(
                        ImageReader(generate_chart_image(fig1)),
                        cost_x + padding_inner / 2,
                        current_y - donut_height + padding_inner / 2,
                        width=donut_width - padding_inner,
                        height=donut_height - padding_inner
                    )

                    revenue_x = PAGE_WIDTH - RIGHT_MARGIN - donut_width
                    c.setStrokeColor(colors.lightgrey)
                    c.setLineWidth(1)
                    c.roundRect(revenue_x, current_y - donut_height, donut_width, donut_height, radius=8, fill=0, stroke=1)
                    fig2 = draw_donut_chart(top_ad_revenue.values, top_ad_revenue.index, "")
                    c.drawImage(
                        ImageReader(generate_chart_image(fig2)),
                        revenue_x + padding_inner / 2,
                        current_y - donut_height + padding_inner / 2,
                        width=donut_width - padding_inner,
                        height=donut_height - padding_inner
                    )
                    current_y -= (donut_height + 60)

                    roas_width, roas_height = 700, 300
                    roas_x = (PAGE_WIDTH - roas_width) / 2
                    c.roundRect(roas_x, current_y - roas_height, roas_width, roas_height, radius=8, fill=0, stroke=1)
                    fig3 = draw_roas_split_bar_chart(top_ad_roas)
                    c.drawImage(ImageReader(generate_chart_image(fig3)), roas_x, current_y - roas_height, width=roas_width, height=roas_height)
                    current_y -= (roas_height + 60)

                    try:
                        cost_chart = generate_cost_by_adset_chart(ad_df)
                        c.setFont("Helvetica-Bold", 14)
                        c.setFillColor(colors.black)
                        c.drawCentredString(PAGE_WIDTH / 2, current_y, "Cost by Ads")
                        current_y -= 20
                        
                        chart_width_line = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
                        chart_height_line = 480
                        c.drawImage(ImageReader(cost_chart[1]), LEFT_MARGIN, current_y - chart_height_line, width=chart_width_line, height=chart_height_line)
                        current_y -= (chart_height_line + 60)
                    except Exception as e:
                        logger.warning(f"⚠️ Error rendering Cost by Ads chart: {str(e)}")
                        current_y -= 50

                    try:
                        revenue_chart = generate_revenue_by_adset_chart(ad_df)
                        c.setFont("Helvetica-Bold", 14)
                        c.drawCentredString(PAGE_WIDTH / 2, current_y, "Revenue by Ads")
                        current_y -= 20
                        
                        c.drawImage(ImageReader(revenue_chart[1]), LEFT_MARGIN, current_y - chart_height_line, width=chart_width_line, height=chart_height_line)
                        current_y -= (chart_height_line + 60)
                    except Exception as e:
                        logger.warning(f"⚠️ Error rendering Revenue by Ads chart: {str(e)}")
                        current_y -= 50

                    try:
                        summary_text = run_async_in_thread(generate_ad_summary(ad_df, currency_symbol))
                        logger.info("📄 Ad LLM Summary Generated")

                        clean_text = re.sub(r"[*#]", "", summary_text).strip()
                        clean_text = re.sub(r"\s{2,}", " ", clean_text)
                        
                        styles = getSampleStyleSheet()
                        styleN = styles["Normal"]
                        styleN.fontName = "DejaVuSans" if currency_symbol == "₹" else "Helvetica"
                        styleN.fontSize = 11
                        styleN.leading = 14
                        styleN.textColor = colors.HexColor("#333333")

                        p = Paragraph(clean_text, styleN)
                        p_width, p_height = p.wrap(PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN, PAGE_HEIGHT)
                        
                        current_y -= (p_height + 20)
                        p.drawOn(c, LEFT_MARGIN, current_y)
                                
                    except Exception as e:
                        logger.warning(f"⚠️ Ad LLM Summary generation failed: {str(e)}")
                        c.setFont("Helvetica", 12)
                        c.setFillColor(colors.red)
                        c.drawString(LEFT_MARGIN, current_y - 50, f"⚠️ Unable to generate ad summary: {str(e)}")
                        current_y -= 100
                else:
                    c.setFont("Helvetica", 12)
                    c.drawCentredString(PAGE_WIDTH / 2, current_y - 50, "No ad data available for detailed analysis.")
                    current_y -= 100

                draw_footer_cta(c)

            elif section_title == "AD FATIGUE ANALYSIS":
                # ( ... Your existing Ad Fatigue Analysis content - untouched ... )
                adjust_page_height(c, section)
                draw_header(c)
                current_y = PAGE_HEIGHT - TOP_MARGIN - 30

                c.setFont("Helvetica-Bold", 16)
                c.setFillColor(colors.black)
                c.drawCentredString(PAGE_WIDTH / 2, current_y, "Ad Fatigue Analysis")
                current_y -= 40

                df = full_ad_insights_df.copy()
                logger.info(f"📊 Columns in DataFrame for Ad Fatigue: {list(df.columns)}")
                logger.info(f"📊 First 5 rows for Ad Fatigue:\n{df.head(5).to_string()}")
                
                if 'impressions' not in df.columns:
                    df['impressions'] = 0
                if 'reach' not in df.columns:
                    df['reach'] = df['impressions']
                    logger.warning("⚠️ 'reach' column missing in Ad Fatigue, using 'impressions' as fallback.")

                df['frequency'] = df['impressions'] / df['reach'].replace(0, 1)
                df['roas'] = df['purchase_value'] / df['spend'].replace(0, 1)
                df['ctr'] = pd.to_numeric(df['ctr'], errors='coerce').fillna(0)
                
                if not df.empty:
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

                    summary_table = Table(table_data, repeatRows=1, colWidths=[150, 150, 170, 70, 50, 60, 60, 60, 40, 60])
                    summary_table.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                        ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans" if currency_symbol == "₹" else "Helvetica"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ]))
                    table_height_actual = summary_table._calcHeight(summary_table._argW)
                    table_y_pos = current_y - table_height_actual - 20
                    summary_table.wrapOn(c, PAGE_WIDTH, PAGE_HEIGHT)
                    summary_table.drawOn(c, LEFT_MARGIN, table_y_pos)
                    current_y = table_y_pos - 40

                    donut_width = 410
                    donut_height = 410

                    if split_charts and len(split_charts) > 0:
                        img_cost = ImageReader(split_charts[0][1])
                        c.drawImage(img_cost, LEFT_MARGIN, current_y - donut_height, width=donut_width, height=donut_height)
                    else:
                        logger.warning("⚠️ Cost Split chart data not available for Ad Fatigue.")

                    if split_charts and len(split_charts) > 1:
                        img_revenue = ImageReader(split_charts[1][1])
                        c.drawImage(img_revenue, PAGE_WIDTH - RIGHT_MARGIN - donut_width, current_y - donut_height, width=donut_width, height=donut_height)
                    else:
                        logger.warning("⚠️ Revenue Split chart data not available for Ad Fatigue.")
                    current_y -= (donut_height + 40)

                    if split_charts and len(split_charts) > 2:
                        img_roas = ImageReader(split_charts[2][1])
                        roas_width = 740
                        roas_height = 320
                        c.drawImage(img_roas, (PAGE_WIDTH - roas_width) / 2, current_y - roas_height, width=roas_width, height=roas_height)
                    else:
                        logger.warning("⚠️ ROAS Split chart data not available for Ad Fatigue.")
                    current_y -= (roas_height + 40)

                    try:
                        freq_chart = generate_frequency_over_time_chart(df)
                        c.setFont("Helvetica-Bold", 14)
                        c.drawCentredString(PAGE_WIDTH / 2, current_y, "Frequency Over Time")
                        current_y -= 20
                        
                        chart_width_line = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
                        chart_height_line = 420
                        c.drawImage(ImageReader(freq_chart[1]), LEFT_MARGIN, current_y - chart_height_line, width=chart_width_line, height=chart_height_line)
                        current_y -= (chart_height_line + 40)
                    except Exception as e:
                        logger.warning(f"⚠️ Error rendering Frequency Over Time chart: {str(e)}")
                        current_y -= 50

                    try:
                        cpm_chart = generate_cpm_over_time_chart(df)
                        c.setFont("Helvetica-Bold", 14)
                        c.drawCentredString(PAGE_WIDTH / 2, current_y, "CPM Over Time")
                        current_y -= 20
                        
                        c.drawImage(ImageReader(cpm_chart[1]), LEFT_MARGIN, current_y - chart_height_line, width=chart_width_line, height=chart_height_line)
                        current_y -= (chart_height_line + 40)
                    except Exception as e:
                        logger.warning(f"⚠️ Error rendering CPM Over Time chart: {str(e)}")
                        current_y -= 50
                    
                    try:
                        summary_text = run_async_in_thread(generate_ad_fatigue_summary(full_ad_insights_df, currency_symbol))
                        logger.info("📄 Ad Fatigue LLM Summary Generated")

                        clean_text = re.sub(r"[*#]", "", summary_text).strip()
                        clean_text = re.sub(r"\s{2,}", " ", clean_text)
                        
                        styles = getSampleStyleSheet()
                        styleN = styles["Normal"]
                        styleN.fontName = "DejaVuSans" if currency_symbol == "₹" else "Helvetica"
                        styleN.fontSize = 11
                        styleN.leading = 14
                        styleN.textColor = colors.HexColor("#333333")

                        p = Paragraph(clean_text, styleN)
                        p_width, p_height = p.wrap(PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN, PAGE_HEIGHT)
                        
                        current_y -= (p_height + 20)
                        p.drawOn(c, LEFT_MARGIN, current_y)
                                
                    except Exception as e:
                        logger.warning(f"⚠️ Ad Fatigue LLM Summary generation failed: {str(e)}")
                        c.setFont("Helvetica", 12)
                        c.setFillColor(colors.red)
                        c.drawString(LEFT_MARGIN, current_y - 50, f"⚠️ Unable to generate ad fatigue summary: {str(e)}")
                        current_y -= 100
                else:
                    c.setFont("Helvetica", 12)
                    c.drawCentredString(PAGE_WIDTH / 2, current_y - 50, "No ad fatigue data available for analysis.")
                    current_y -= 100

                draw_footer_cta(c)


            # ─────────────────────────────────────────────────────────────
            # FIX STARTS HERE: Demographic Performance
            # ─────────────────────────────────────────────────────────────
            elif section_title == "DEMOGRAPHIC PERFORMANCE":
                # Page break is handled by `if i > 0:` at the top.
                adjust_page_height(c, section) # This ensures PAGE_HEIGHT is 3000
                draw_header(c)
                current_y = PAGE_HEIGHT - TOP_MARGIN - 30 # Reset Y for this page

                c.setFont("Helvetica-Bold", 16)
                c.setFillColor(colors.black)
                c.drawCentredString(PAGE_WIDTH / 2, current_y, "Demographic Performance")
                current_y -= 40 # Space after title

                if demographic_df is not None and not demographic_df.empty and \
                   'age' in demographic_df.columns and 'gender' in demographic_df.columns and \
                   demographic_df['spend'].sum() > 0:
                    logger.info("Proceeding with Demographic Performance section as data is valid.")

                    # Ensure numeric columns are present and cleaned
                    for col in ['spend', 'purchase_value', 'purchases']:
                        if col not in demographic_df.columns:
                            demographic_df[col] = 0
                        else:
                            demographic_df[col] = pd.to_numeric(demographic_df[col], errors='coerce').fillna(0)

                    # Recalculate ROAS and CPA after ensuring numeric columns
                    demographic_df['roas'] = demographic_df['purchase_value'] / demographic_df['spend'].replace(0, 1)
                    demographic_df['cpa'] = demographic_df['spend'] / demographic_df['purchases'].replace(0, 1)

                    demographic_grouped = demographic_df.groupby(['age', 'gender']).agg({
                        'spend': 'sum',
                        'purchases': 'sum',
                        'roas': 'mean',
                        'cpa': 'mean'
                    }).reset_index()

                    demographic_grouped.rename(columns={
                        'age': 'Age',
                        'gender': 'Gender',
                        'spend': 'Amount Spent',
                        'purchases': 'Purchases',
                        'roas': 'ROAS',
                        'cpa': 'CPA'
                    }, inplace=True)

                    demographic_grouped['Amount Spent'] = demographic_grouped['Amount Spent'].apply(lambda x: f"{currency_symbol}{x:,.2f}")
                    demographic_grouped['CPA'] = demographic_grouped['CPA'].apply(lambda x: f"{currency_symbol}{x:,.2f}")
                    demographic_grouped['ROAS'] = demographic_grouped['ROAS'].round(2)

                    # 📋 Draw Table
                    table_data = [demographic_grouped.columns.tolist()] + demographic_grouped.values.tolist()
                    table_col_widths = [100, 80, 100, 80, 80, 80]
                    table = Table(table_data, colWidths=table_col_widths)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('FONTNAME', (0, 1), (-1, -1), "DejaVuSans" if currency_symbol == "₹" else "Helvetica"),
                        ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ]))

                    table_width, table_height = table.wrapOn(c, PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN, PAGE_HEIGHT)
                    table_x = LEFT_MARGIN + (PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN - table_width) / 2
                    
                    # Draw table using current_y
                    table.drawOn(c, table_x, current_y - table_height)
                    current_y -= (table_height + 40) # Update Y position below table

                    # --- Draw Demographic Charts ---
                    chart_width = 300
                    chart_height = 250
                    chart_padding_x = 50
                    chart_padding_y = 30

                    # Row 1: Cost, Revenue by Age
                    # Ensure demographic_grouped has the necessary columns for charts (e.g., 'Amount Spent' not 'spend')
                    # The rename() call above already handles this, so use the renamed columns.
                    if not demographic_grouped.empty and {'Age', 'Amount Spent', 'Purchase Value', 'Purchases', 'ROAS'}.issubset(demographic_grouped.columns):
                        try:
                            cost_age_chart_buf = generate_cost_split_by_age_chart(demographic_grouped)
                            c.drawImage(ImageReader(cost_age_chart_buf), LEFT_MARGIN, current_y - chart_height, width=chart_width, height=chart_height, preserveAspectRatio=True)

                            revenue_age_chart_buf = generate_revenue_split_by_age_chart(demographic_grouped)
                            c.drawImage(ImageReader(revenue_age_chart_buf), LEFT_MARGIN + chart_width + chart_padding_x, current_y - chart_height, width=chart_width, height=chart_height, preserveAspectRatio=True)

                            current_y -= (chart_height + chart_padding_y)

                            # Row 2: ROAS by Age & Cost, Revenue by Gender
                            roas_age_chart_buf = generate_roas_split_by_age_chart(demographic_grouped)
                            c.drawImage(ImageReader(roas_age_chart_buf), LEFT_MARGIN, current_y - chart_height, width=chart_width, height=chart_height, preserveAspectRatio=True)

                            cost_gender_chart_buf = generate_cost_split_by_gender_chart(demographic_grouped)
                            c.drawImage(ImageReader(cost_gender_chart_buf), LEFT_MARGIN + chart_width + chart_padding_x, current_y - chart_height, width=chart_width, height=chart_height, preserveAspectRatio=True)

                            current_y -= (chart_height + chart_padding_y)

                            # Row 3: Revenue by Gender & ROAS by Gender
                            revenue_gender_chart_buf = generate_revenue_split_by_gender_chart(demographic_grouped)
                            c.drawImage(ImageReader(revenue_gender_chart_buf), LEFT_MARGIN, current_y - chart_height, width=chart_width, height=chart_height, preserveAspectRatio=True)

                            roas_gender_chart_buf = generate_roas_split_by_gender_chart(demographic_grouped)
                            c.drawImage(ImageReader(roas_gender_chart_buf), LEFT_MARGIN + chart_width + chart_padding_x, current_y - chart_height, width=chart_width, height=chart_height, preserveAspectRatio=True)

                            current_y -= (chart_height + chart_padding_y)

                        except Exception as e:
                            logger.error(f"Error drawing demographic charts: {e}")
                            c.setFont("Helvetica", 12)
                            c.setFillColor(colors.red)
                            c.drawString(LEFT_MARGIN, current_y - 50, f"⚠️ Error generating demographic charts: {str(e)}")
                            current_y -= 100

                    # 📝 LLM Summary - Dynamic
                    try:
                        summary_text = run_async_in_thread(build_demographic_summary_prompt(demographic_grouped, currency_symbol))
                        logger.info("Demographic LLM Summary Generated.")

                        clean_text = re.sub(r"[*#]", "", summary_text).strip()
                        clean_text = re.sub(r"\s{2,}", " ", clean_text)

                        styles = getSampleStyleSheet()
                        styleN = styles["Normal"]
                        styleN.fontName = "DejaVuSans" if currency_symbol == "₹" else "Helvetica"
                        styleN.fontSize = 11
                        styleN.leading = 14
                        styleN.textColor = colors.HexColor("#333333")

                        p = Paragraph(clean_text, styleN)
                        p_width, p_height = p.wrap(PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN, PAGE_HEIGHT)
                        
                        current_y -= (p_height + 40)
                        p.drawOn(c, LEFT_MARGIN, current_y)

                    except Exception as e:
                        logger.error(f"Demographic LLM Summary generation failed: {e}")
                        c.setFont("Helvetica", 12)
                        c.setFillColor(colors.red)
                        c.drawString(LEFT_MARGIN, current_y - 50, f"⚠️ Unable to generate demographic summary: {str(e)}")
                        current_y -= 100
                else: # This block executes if demographic_df is not valid for processing
                    logger.warning("Demographic data not available or insufficient for detailed analysis. Skipping section.")
                    c.setFont("Helvetica", 14)
                    c.setFillColor(colors.black)
                    c.drawCentredString(PAGE_WIDTH / 2, current_y - 50, "⚠️ Demographic data not available for this account or contains no valid entries.")
                    current_y -= 100

                draw_footer_cta(c)
            # ─────────────────────────────────────────────────────────────
            # FIX ENDS HERE: Demographic Performance
            # ─────────────────────────────────────────────────────────────
            
            # ─────────────────────────────────────────────────────────────
            # Generic Text-Only Sections (EXECUTIVE SUMMARY, etc.)
            # ─────────────────────────────────────────────────────────────
            else: # This 'else' catches EXECUTIVE SUMMARY, ACCOUNT NAMING, TESTING, REMARKETING, RESULTS SETUP
                # This 'else' block will only be hit if a section_title doesn't match any of the explicit elifs.
                # Your log shows "EXECUTIVE SUMMARY", "ACCOUNT NAMING & STRUCTURE", etc. as separate sections.
                # They should now render without the two-half division if this 'else' is correctly structured.

                c.setFont("Helvetica-Bold", 22)
                c.setFillColor(colors.black)
                c.drawCentredString(PAGE_WIDTH / 2, current_y, section_title)
                current_y -= 40 # Space after title

                set_font_with_currency(c, currency_symbol, size=14)
                c.setFillColor(colors.black) 
                
                styles = getSampleStyleSheet()
                styleN = styles["Normal"]
                styleN.fontName = "DejaVuSans" if currency_symbol == "₹" else "Helvetica"
                styleN.fontSize = 11
                styleN.leading = 14
                styleN.textColor = colors.HexColor("#333333")

                p = Paragraph(content, styleN)
                p_width, p_height = p.wrap(PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN, PAGE_HEIGHT)
                
                # Check if content fits on current page. If not, start new page.
                # This check ensures that long text content flows to a new page correctly.
                if current_y - p_height < BOTTOM_MARGIN + 60: # 60 is for footer space
                    c.showPage()
                    adjust_page_height(c, section) # Re-adjust for current section
                    draw_header(c)
                    current_y = PAGE_HEIGHT - TOP_MARGIN - 30 # Reset Y for new page
                    c.setFont("Helvetica-Bold", 22) # Redraw title on new page
                    c.setFillColor(colors.black)
                    c.drawCentredString(PAGE_WIDTH / 2, current_y, section_title)
                    current_y -= 40

                p.drawOn(c, LEFT_MARGIN, current_y - p_height)
                current_y -= (p_height + 20) # Update Y position

                draw_footer_cta(c)


        c.save()
        buffer.seek(0)
        return StreamingResponse(io.BytesIO(buffer.read()), media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=audit_report.pdf"})

    except Exception as e:
        logger.error(f"❌ Error generating PDF: {str(e)}", exc_info=True)
        raise Exception(f"Failed to generate PDF: {str(e)}")