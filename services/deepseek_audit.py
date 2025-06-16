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



def generate_pdf_report(content: str) -> StreamingResponse:
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 50
    y = height - margin

    # Dimensions
    line_x = margin + 50
    text_start_x = line_x + 20
    icon_size = 40

    # === DRAW ICON (placeholder circle instead of image) ===
    icon_center_x = margin + icon_size // 2
    icon_center_y = y - icon_size // 2 + 10
    p.setFillColorRGB(0.8, 1, 0.6)  # Light green
    p.circle(icon_center_x, icon_center_y, icon_size // 2, fill=True, stroke=False)

    # === DRAW BLUE VERTICAL LINE ===
    p.setFillColor(colors.blue)
    p.rect(line_x, y - 60, 4, 50, fill=True, stroke=False)

    # === TITLE ===
    p.setFont("Helvetica-Bold", 18)
    p.setFillColor(colors.black)
    p.drawString(text_start_x, y, "EXECUTIVE SUMMARY")
    y -= 30

    # === PARAGRAPH TEXT ===
    summary_paragraph = ""
    lines = content.splitlines()
    i = 0

    # Collect the summary paragraph
    while i < len(lines):
        if "EXECUTIVE SUMMARY" in lines[i].upper():
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("1."):
                summary_paragraph += lines[i].strip() + " "
                i += 1
            break
        i += 1

    # Wrap paragraph and print
    wrapped_lines = simpleSplit(summary_paragraph.strip(), "Helvetica", 11, width - text_start_x - margin)
    p.setFont("Helvetica", 11)
    for line in wrapped_lines:
        p.drawString(text_start_x, y, line)
        y -= 16
        if y < 50:
            p.showPage()
            y = height - margin

    # === Continue Rest of the Audit Content ===
    p.setFont("Helvetica-Bold", 13)
    p.drawString(margin, y - 20, "DETAILED AUDIT")
    y -= 40
    p.setFont("Helvetica", 11)

    while i < len(lines):
        line = lines[i].strip()
        clean_line = line.replace("**", "").replace("#", "").replace("*", "")
        p.drawString(margin, y, clean_line)
        y -= 18
        if y < 50:
            p.showPage()
            y = height - margin
        i += 1

    p.save()
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/pdf", headers={
        "Content-Disposition": "attachment; filename=audit_report.pdf"
    })
