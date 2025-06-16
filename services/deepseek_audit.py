import httpx
import os
from datetime import datetime, timedelta
import traceback
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io
from fastapi.responses import StreamingResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.utils import simpleSplit
import io
from fastapi.responses import StreamingResponse

DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

SYSTEM_PROMPT = """
You are a social media analytics expert. Given Facebook Insights data for a business page, provide:
1. Spend trend over the past 2 months
2. Top 5 campaigns by spend
3. Impressions, clicks, and top actions with their values
Make the output concise, readable, and insightful.
"""

async def fetch_facebook_insights(page_id: str, page_token: str):
    base_url = f"https://graph.facebook.com/v18.0/{page_id}/insights"
    since = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    until = datetime.now().strftime("%Y-%m-%d")

    # params = {
    #     "metric": ",".join([
    #         "page_impressions",
    #         "page_engaged_users",
    #         "page_views_total",
    #         "page_post_engagements"
            
    #     ]),
    #     "since": since,
    #     "until": until,
    #     "access_token": page_token
    # }
    params = {
    "metric": "page_impressions",  # safest metric
    "since": since,
    "until": until,
    "access_token": page_token
    }


    async with httpx.AsyncClient() as client:
        print("📡 Request URL:", client.build_request("GET", base_url, params=params).url)
        resp = await client.get(base_url, params=params)
        print("🔎 Metrics sent:", params["metric"])

        print("🔍 Facebook Insights API Raw Response:")
        print(resp.text)  # Debugging line to see the raw response
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            print(" Facebook API returned error:", e.response.text)
            raise Exception(f"Facebook API Error: {e.response.text}")
        
        return resp.json()

async def fetch_ad_insights(page_token: str):
    url = f"https://graph.facebook.com/v18.0/me/adaccounts"
    async with httpx.AsyncClient() as client:
        try:
            # Fetch ad accounts associated with the page token
            acc_resp = await client.get(url, params={"access_token": page_token})
            acc_resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            print("Could not fetch ad accounts — skipping ad insights.")
            return []
        accounts = acc_resp.json().get("data", [])

        insights_data = []
        for acc in accounts:
            ad_url = f"https://graph.facebook.com/v18.0/{acc['id']}/insights"
            ad_params = {
                "fields": "campaign_name,spend,impressions,clicks,cpc,ctr",
                "date_preset": "last_60_days",
                "access_token": page_token
            }
            insights_resp = await client.get(ad_url, params=ad_params)
            if insights_resp.status_code == 200:
                insights_data.extend(insights_resp.json().get("data", []))
        return insights_data

async def generate_audit(page_id: str, page_token: str):
    try:
        page_data = await fetch_facebook_insights(page_id, page_token)
        ad_data = await fetch_ad_insights(page_token)

        combined_data = {
            "page_insights": page_data,
            "ad_insights": ad_data
        }

        prompt = f"Analyze the following Facebook data:\n{combined_data}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(
                DEEPSEEK_API_URL,
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ]
                }
            )
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("❌ Exception Traceback:")
        traceback.print_exc()
        return f"Audit generation failed: {str(e) or repr(e)}"
        #return f"Audit generation failed: {str(e)}"

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.utils import simpleSplit
import io
from fastapi.responses import StreamingResponse

def generate_pdf_report(content: str) -> StreamingResponse:
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 50
    y = height - margin

    # DESIGN LAYOUT VARIABLES
    icon_size = 40
    icon_x = margin
    icon_y = y - icon_size
    line_x = icon_x + icon_size + 10
    line_y = icon_y
    line_width = 5
    line_height = icon_size
    text_x = line_x + 10
    text_width = width - text_x - margin

    lines = content.splitlines()
    summary_paragraph = ""
    found_summary = False
    summary_end_index = 0

    # Extract EXECUTIVE SUMMARY paragraph
    for i in range(len(lines)):
        line = lines[i].strip()
        if "EXECUTIVE SUMMARY" in line.upper():
            found_summary = True
            continue
        if found_summary:
            if line.startswith("1.") or line.startswith("Key Insight") or line == "":
                summary_end_index = i
                break
            summary_paragraph += line + " "

    if not summary_paragraph.strip():
        summary_paragraph = "[No summary content generated by LLM]"

    # Draw icon (placeholder circle)
    p.setFillColorRGB(0.8, 1, 0.6)  # light green
    p.circle(icon_x + icon_size // 2, icon_y + icon_size // 2, icon_size // 2, fill=True, stroke=False)

    # Draw blue vertical line
    p.setFillColor(colors.blue)
    p.rect(line_x, line_y, line_width, line_height, fill=True, stroke=False)

    # Draw title
    p.setFont("Helvetica-Bold", 18)
    p.setFillColor(colors.black)
    p.drawString(text_x, y, "EXECUTIVE SUMMARY")
    y -= 30

    # Draw paragraph
    p.setFont("Helvetica", 11)
    wrapped_lines = simpleSplit(summary_paragraph.strip(), "Helvetica", 11, text_width)
    for line in wrapped_lines:
        p.drawString(text_x, y, line)
        y -= 16
        if y < 50:
            p.showPage()
            y = height - margin

    # Section separator
    y -= 20
    p.setFont("Helvetica-Bold", 13)
    p.drawString(margin, y, "DETAILED AUDIT")
    y -= 25
    p.setFont("Helvetica", 11)

    # Draw rest of the content after EXECUTIVE SUMMARY
    for i in range(summary_end_index, len(lines)):
        clean_line = lines[i].replace("**", "").replace("#", "").replace("*", "").strip()
        if not clean_line:
            continue
        wrapped = simpleSplit(clean_line, "Helvetica", 11, width - 2 * margin)
        for wrap_line in wrapped:
            p.drawString(margin, y, wrap_line)
            y -= 16
            if y < 50:
                p.showPage()
                y = height - margin

    # Finalize
    p.save()
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/pdf", headers={
        "Content-Disposition": "attachment; filename=audit_report.pdf"
    })
