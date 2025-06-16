# services/deepseek_audit.py
# services/deepseek_audit.py
import httpx
import os
from datetime import datetime, timedelta
from fastapi.responses import StreamingResponse
from services.prompts import EXECUTIVE_SUMMARY_PROMPT
from services.generate_pdf import generate_pdf_report

DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

async def fetch_facebook_insights(page_id: str, page_token: str):
    """Fetch Facebook page insights"""
    try:
        base_url = f"https://graph.facebook.com/v18.0/{page_id}/insights"
        since = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        until = datetime.now().strftime("%Y-%m-%d")

        params = {
            "metric": "page_impressions",
            "since": since,
            "until": until,
            "access_token": page_token
        }

        async with httpx.AsyncClient() as client:
            resp = await client.get(base_url, params=params)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        print(f"❌ Error fetching Facebook insights: {str(e)}")
        return {"data": [], "error": str(e)}


async def fetch_ad_insights(page_token: str):
    """Fetch Facebook ad insights"""
    try:
        url = f"https://graph.facebook.com/v18.0/me/adaccounts"
        async with httpx.AsyncClient() as client:
            acc_resp = await client.get(url, params={"access_token": page_token})
            acc_resp.raise_for_status()
            accounts = acc_resp.json().get("data", [])

            insights_data = []
            for acc in accounts:
                try:
                    ad_url = f"https://graph.facebook.com/v18.0/{acc['id']}/insights"
                    ad_params = {
                        "fields": "campaign_name,spend,impressions,clicks,cpc,ctr",
                        "date_preset": "last_60_days",
                        "access_token": page_token
                    }
                    insights_resp = await client.get(ad_url, params=ad_params)
                    if insights_resp.status_code == 200:
                        insights_data.extend(insights_resp.json().get("data", []))
                    else:
                        print(f"⚠️ Warning: Failed to fetch insights for account {acc['id']}")
                except Exception as e:
                    print(f"⚠️ Warning: Error fetching insights for account {acc.get('id', 'unknown')}: {str(e)}")
                    continue
            
            return insights_data
    except Exception as e:
        print(f"❌ Error fetching ad insights: {str(e)}")
        return []


async def generate_audit(page_id: str, page_token: str):
    """Generate audit report and return PDF"""
    try:
        print("🔄 Starting audit generation...")
        
        # Validate environment variables
        if not DEEPSEEK_API_URL or not DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_URL and DEEPSEEK_API_KEY environment variables must be set")
        
        # Fetch data from Facebook
        print("📊 Fetching Facebook data...")
        page_data = await fetch_facebook_insights(page_id, page_token)
        ad_data = await fetch_ad_insights(page_token)

        combined_data = {
            "page_insights": page_data,
            "ad_insights": ad_data
        }

        # Call LLM for analysis
        print("🤖 Generating analysis with DeepSeek...")
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
                        {"role": "system", "content": EXECUTIVE_SUMMARY_PROMPT},
                        {"role": "user", "content": prompt}
                    ]
                }
            )
            res.raise_for_status()
            
            response_data = res.json()
            if "choices" not in response_data or not response_data["choices"]:
                raise ValueError("Invalid response from DeepSeek API")
                
            summary = response_data["choices"][0]["message"]["content"]
            print("✅ Summary generated successfully")

        # Generate PDF
        print("📄 Generating PDF report...")
        pdf_response = generate_pdf_report("EXECUTIVE SUMMARY", summary)
        print("✅ PDF generated successfully")
        
        return pdf_response
        
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error from external API: {e.response.status_code} - {e.response.text}"
        print(f"❌ {error_msg}")
        raise Exception(error_msg)
    except httpx.RequestError as e:
        error_msg = f"Request error: {str(e)}"
        print(f"❌ {error_msg}")
        raise Exception(error_msg)
    except Exception as e:
        error_msg = f"Error generating audit: {str(e)}"
        print(f"❌ {error_msg}")
        raise Exception(error_msg)