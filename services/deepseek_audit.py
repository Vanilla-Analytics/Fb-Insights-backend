import httpx
import os
from datetime import datetime, timedelta
import traceback
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
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
    x, y = 50, height - 50

    for line in content.splitlines():
        line = line.replace("**", "").replace("#", "").replace("*", "").strip()
        if line.lower().startswith("facebook") or line.endswith(":"):
            p.setFont("Helvetica-Bold", 12)
        else:
            p.setFont("Helvetica", 11)

        p.drawString(x, y, line)
        y -= 18
        if y < 50:
            p.showPage()
            y = height - 50

    p.save()
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/pdf", headers={
        "Content-Disposition": "attachment; filename=audit_report.pdf"
    })
