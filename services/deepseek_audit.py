# services/deepseek_audit.py
import httpx
import os
import requests
from datetime import datetime, timedelta
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from services.prompts import EXECUTIVE_SUMMARY_PROMPT, ACCOUNT_NAMING_STRUCTURE_PROMPT
from services.prompts import TESTING_ACTIVITY_PROMPT
from services.prompts import REMARKETING_ACTIVITY_PROMPT
from services.prompts import RESULTS_SETUP_PROMPT
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from services.generate_pdf import generate_pdf_report
import json

DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

def generate_chart_1(ad_insights_df):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(18, 6), dpi=200)

    # Fill missing values
    ad_insights_df['purchase_value'] = ad_insights_df['purchase_value'].fillna(0)
    ad_insights_df['spend'] = ad_insights_df['spend'].fillna(0)

    # Twin axis
    ax2 = ax1.twinx()

    # ✅ Green Bars (use ax2 so they share y-axis with spend)
    ax2.bar(
        ad_insights_df["date"],
        ad_insights_df["purchase_value"],
        color="#B2FF59",           # Bright green
        edgecolor="#76FF03",
        width=0.6,
        label="Purchase Conversion Value",
        alpha=0.9,
        zorder=2
    )

    # ✅ Magenta Line (on same ax2 to align with green bars)
    ax2.plot(
        ad_insights_df["date"],
        ad_insights_df["spend"],
        color="magenta",
        marker="o",
        label="Amount Spent",
        linewidth=2.5,
        zorder=3
    )

    # Labels
    ax1.set_ylabel("Purchases", color="#4CAF50", fontsize=12)
    ax2.set_ylabel("Amount Spent", color="magenta", fontsize=12)

    ax1.tick_params(axis='y', labelcolor="#4CAF50", labelsize=12)
    ax2.tick_params(axis='y', labelcolor="magenta", labelsize=12)

    # X-axis formatting
    ax1.set_xticks(ad_insights_df["date"])
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax1.tick_params(axis='x', rotation=45, labelsize=10)

    # Grid
    ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    return fig




def generate_chart_image(fig):
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=200)  # Removed bbox_inches
    buf.seek(0)
    plt.close(fig)
    return buf
    # buf = BytesIO()
    # fig.savefig(buf, format='png', bbox_inches='tight')
    # buf.seek(0)
    # encoded = base64.b64encode(buf.read()).decode('utf-8')
    # plt.close(fig)
    # buf = BytesIO()
    # fig.savefig(buf, format='png', bbox_inches='tight')
    # buf.seek(0)
    # plt.close(fig)
    # return buf


def generate_key_metrics_section(ad_insights_df, currency_symbol="$"):
    
    # if 'account_currency' not in ad_insights_df.columns:
    #     currency_symbol = "$"  # Default to USD
    # else:
    #     currency_symbol = ad_insights_df['account_currency'].mode()[0] if not ad_insights_df['account_currency'].mode().empty else "$"
    #     currency_symbol = "₹" if currency_symbol == "INR" else "$"

    if ad_insights_df.empty or len(ad_insights_df) < 2:
        print("⚠️ Not enough data to generate charts.")
        return "No data available for Key Metrics.", []

    metrics_summary = {
        "Amount Spent": f"{currency_symbol}{ad_insights_df['spend'].sum():,.2f}",
        "Purchases": int(ad_insights_df['purchases'].sum()),
        "Purchase Value": f"{currency_symbol}{ad_insights_df['purchase_value'].sum():,.2f}",
        "ROAS": round(ad_insights_df['roas'].mean(), 2),
        "CPA": f"{currency_symbol}{ad_insights_df['cpa'].mean():.2f}",
        "Cost/Result": f"{currency_symbol}{ad_insights_df['cpa'].mean():.2f}",
        "Impressions": int(ad_insights_df['impressions'].sum()),
        "CPM": f"{currency_symbol}{(ad_insights_df['spend'].sum() / ad_insights_df['impressions'].sum() * 1000):.2f}",
        "Link Clicks": int(ad_insights_df['clicks'].sum()),
        "Link CPC": f"{currency_symbol}{ad_insights_df['cpc'].mean():.2f}",
        "CTR (link)": f"{ad_insights_df['ctr'].mean():.2%}"
    }
    

    summary_text = "\n".join([f"{k}: {v}" for k, v in metrics_summary.items()])

    # Charts
    chart_imgs = []

    # fig1, ax1 = plt.subplots(figsize=(12, 4))  # Wider chart
    # ax1.bar(ad_insights_df['date'], ad_insights_df['purchase_value'], color='lightgreen', label='Purchase Value')

    # ax2 = ax1.twinx()
    # ax2.plot(ad_insights_df['date'], ad_insights_df['spend'], color='magenta', marker='o', label='Amount Spent')

    # ax1.set_title("Amount Spent vs Purchase Conversion Value")
    # ax1.xaxis.set_major_locator(MaxNLocator(nbins=10))  # Limit number of X-axis labels
    # fig1.autofmt_xdate(rotation=45)  # Rotate X-axis dates to avoid overlap

    #chart_imgs = []

    # Chart 1: Amount Spent vs Purchase Conversion Value
    
    fig1 = generate_chart_1(ad_insights_df)
    chart_imgs.append(("", generate_chart_image(fig1)))



    #chart_imgs.append(("Amount Spent vs Purchase Conversion Value", generate_chart_image(fig1)))


    # Chart 2: Purchases vs ROAS
    fig2, ax3 = plt.subplots(figsize=(12, 4))
    ax3.bar(ad_insights_df['date'], ad_insights_df['purchases'], color='darkblue', label='Purchases')
    ax4 = ax3.twinx()
    ax4.plot(ad_insights_df['date'], ad_insights_df['roas'], color='magenta', marker='o', label='ROAS')
    ax3.set_title("Purchases vs ROAS")
    ax3.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax3.tick_params(axis='x', rotation=45, labelsize=10)
    # ax3.set_ylabel("Purchases")
    # ax4.set_ylabel("ROAS")  

    chart_imgs.append(("Purchases vs ROAS", generate_chart_image(fig2)))

    # Chart 3: CPA vs Link CPC (Dual Y-Axis)
    fig3, ax5 = plt.subplots(figsize=(12, 4))
    ax5.bar(ad_insights_df['date'], ad_insights_df['cpa'], color='blue', label='CPA')
    #ax5.plot(ad_insights_df['date'], ad_insights_df['cpa'], color='blue', label='CPA')
    ax6 = ax5.twinx()
    ax6.plot(ad_insights_df['date'], ad_insights_df['cpc'], color='pink', label='Link CPC')
    ax5.set_title("CPA vs Link CPC")
    ax5.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax5.tick_params(axis='x', rotation=45, labelsize=10)

    chart_imgs.append(("CPA vs Link CPC", generate_chart_image(fig3)))
    # ax5.set_ylabel("CPA")
    # ax6.set_ylabel("Link CPC")
    # ax5.set_title("CPA vs Link CPC")
    # chart_imgs.append(("CPA vs Link CPC", generate_chart_image(fig3)))
    


    # Chart 4: Click to Conversion vs CTR
    fig4, ax7 = plt.subplots(figsize=(12, 4))
    #ax7.plot(ad_insights_df['date'], ad_insights_df['click_to_conversion'], color='pink', label='Click to Conversion')
    ax7.bar(ad_insights_df['date'], ad_insights_df['click_to_conversion'], color='pink', label='Click to Conversion')
    ax8 = ax7.twinx()
    ax8.plot(ad_insights_df['date'], ad_insights_df['ctr'], color='darkblue', label='CTR')
    ax7.set_title("Click to Conversion vs CTR")
    ax7.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax7.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax7.tick_params(axis='x', rotation=45, labelsize=10)

    chart_imgs.append(("Click to Conversion vs CTR", generate_chart_image(fig4)))
    # ax7.set_ylabel("Click to Conversion")
    # ax8.set_ylabel("CTR")
    # ax7.set_title("Click to Conversion vs CTR")
    # chart_imgs.append(("Click to Conversion vs CTR", generate_chart_image(fig4)))

    # Table summary
    table_html = ad_insights_df.to_html(index=False, border=0)

    # Combined content
    content = f"""

    **Key Metrics**

    {summary_text}

    **Last 30 Days Trend Section**

    The following section presents daily trend of the Key Metrics Identified in the previous section. This helps the business analyse the daily variance in the business KPIs and also helps in correlating how one metric affects the others.

    {table_html}
    """

    return {
        "title": "KEY METRICS",
        "content": content,
        "charts": chart_imgs
    }

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
        #url = f"https://graph.facebook.com/v18.0/me/adaccounts"
        url = f"https://graph.facebook.com/v18.0/me/adaccounts?fields=account_id,account_currency"
        async with httpx.AsyncClient() as client:
            acc_resp = await client.get(url, params={"access_token": page_token})
            acc_resp.raise_for_status()
            # if acc_resp.status_code != 200:
            #     print("⚠️ Facebook API Error:", acc_resp.text)
            #     acc_resp.raise_for_status()

            accounts = acc_resp.json().get("data", [])
            print("📡 Ad Accounts fetched:", accounts)

            # Map account ID to currency
            # account_currency_map = {
            #     acc["id"]: acc.get("account_currency", "USD") for acc in accounts
            # }

            account_currency_map = {
                str(acc.get("account_id") or acc.get("id")): acc.get("account_currency", "USD")
                for acc in accounts
            }


            # accounts = acc_resp.json().get("data", [])
            # print("📡 Ad Accounts fetched:", accounts)
            # accounts_data = acc_resp.json().get("data", [])
            # account_currency_map = {acc['id']: acc.get('account_currency', 'USD') for acc in accounts_data}

            from datetime import datetime, timedelta

            since = (datetime.today() - timedelta(days=60)).strftime('%Y-%m-%d')
            until = datetime.today().strftime('%Y-%m-%d')
            insights_data = []
            for acc in accounts:
                try:
                    ad_url = f"https://graph.facebook.com/v22.0/{acc['id']}/insights"
                    
                    today = datetime.today()
                    sixty_days_ago = today - timedelta(days=60)
                    ad_params = {
                        #"fields": "campaign_name,adset_name,ad_name,spend,impressions,clicks,cpc,ctr",
                        "fields": "campaign_name,adset_name,ad_name,spend,impressions,clicks,cpc,ctr,actions,action_values",
                        "time_range": json.dumps({
                            "since": sixty_days_ago.strftime("%Y-%m-%d"),
                            "until": today.strftime("%Y-%m-%d")
                        }),
                        #"time_range": {"since": since, "until": until},
                        #"date_preset": "last_60_days",
                        #"date_preset":"maximum",
                        "time_increment": 1,  # 👈 daily breakdown
                        "level": "ad",        # 👈 required to enable daily granularity
                        "access_token": page_token
                    }
                    insights_response = requests.get(ad_url, params=ad_params)
                    
                    print(f"📡 Requesting URL: {ad_url}")
                    print(f"📡 With params: {ad_params}")
                    print(f"📦 Response status: {insights_response.status_code}")
                    print(f"📦 Response body: {insights_response.text}")

                    
                    # if insights_response.status_code == 200:
                    #     ad_results = insights_response.json().get("data", [])
                    #     for ad in ad_results:
                    #         ad["account_currency"] = account_currency_map.get(acc["id"], "USD")
                    #         insights_data.append(ad)
                    # else:
                    #     print(f"⚠️ Warning: Failed to fetch insights for account {acc['id']}")
                    #     print(f"🔍 Status: {insights_response.status_code}, Content: {insights_response.text}")
                    #     print(f"📊 Insights from account {acc['id']}: {len(ad_results)} entries")  # ✅ This is your count message
                    #     for ad in ad_results:
                    #         ad["account_currency"] = account_currency_map.get(acc["id"], "USD")
                    #         insights_data.append(ad)
                        
                    ad_results = insights_response.json().get("data", [])

                    if insights_response.status_code == 200:
                        print(f"📊 Insights from account {acc['id']}: {len(ad_results)} entries")
                    else:
                        print(f"⚠️ Warning: Failed to fetch insights for account {acc['id']}")
                        print(f"🔍 Status: {insights_response.status_code}, Content: {insights_response.text}")

                    for ad in ad_results:

                        ad["account_currency"] = account_currency_map.get(str(acc.get("account_id") or acc.get("id")), "USD")

                        insights_data.append(ad)

                except Exception as e:
                    print(f"⚠️ Warning: Error fetching insights for account {acc.get('id', 'unknown')}: {str(e)}")
                    continue
            
            return insights_data
    except Exception as e:
        print(f"❌ Error fetching ad insights: {str(e)}")
        print(f"🔍 Response: {acc_resp.text}")
        return []


async def generate_llm_content(prompt: str, data: dict) -> str:
    """Generate content using DeepSeek LLM"""
    try:
        data_prompt = f"Analyze the following Facebook data:\n{data}"
        
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
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": data_prompt}
                    ]
                }
            )
            res.raise_for_status()
            
            response_data = res.json()
            if "choices" not in response_data or not response_data["choices"]:
                raise ValueError("Invalid response from DeepSeek API")
                
            return response_data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ Error generating LLM content: {str(e)}")
        return f"Error generating content: {str(e)}"


async def generate_audit(page_id: str, user_token: str, page_token: str):
    """Generate audit report and return PDF"""
    try:
        print("🔄 Starting audit generation...")

        if not DEEPSEEK_API_URL or not DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_URL and DEEPSEEK_API_KEY environment variables must be set")

        print("📊 Fetching Facebook data...")
        page_data = await fetch_facebook_insights(page_id, page_token)
        ad_data = await fetch_ad_insights(user_token)

        # Filter out invalid entries
        ad_data = [d for d in ad_data if isinstance(d, dict) and 'date_start' in d and d.get('date_start')]
        if not ad_data:
            raise ValueError("❌ All ad insights entries are missing 'date_start' — cannot proceed.")
        
        PURCHASE_KEYS = [
            "offsite_conversion.purchase",
            "offsite_conversion.fb_pixel_purchase",
            "offsite_conversion.custom.1408006162945363",
            "purchase"
        ]
        

        # for ad in ad_data:
        #     actions = {d["action_type"]: float(d["value"]) for d in ad.get("actions", []) if "action_type" in d and "value" in d}
        #     values = {d["action_type"]: float(d["value"]) for d in ad.get("action_values", []) if "action_type" in d and "value" in d}
        #     ad["purchases"] = actions.get("purchase", 0)
        #     ad["purchase_value"] = values.get("purchase", 0)
        for ad in ad_data:
            try:
                actions = {d["action_type"]: float(d["value"]) for d in ad.get("actions", []) if "action_type" in d and "value" in d}
                values = {d["action_type"]: float(d["value"]) for d in ad.get("action_values", []) if "action_type" in d and "value" in d}
            except Exception as e:
                print(f"⚠️ Error parsing actions in ad: {e}")
                actions, values = {}, {}

            # Safely sum purchase-related values
            ad["purchases"] = sum(actions.get(k, 0) for k in PURCHASE_KEYS)
            ad["purchase_value"] = sum(values.get(k, 0) for k in PURCHASE_KEYS)



        # Create original DataFrame with date_start intact
        original_df = pd.DataFrame(ad_data)

        if 'date_start' not in original_df.columns:
            raise ValueError("❌ 'date_start' column is missing in ad insights data.")

        original_df = original_df.dropna(subset=['date_start'])
        original_df['date_start'] = original_df['date_start'].astype(str)
        original_df['date'] = pd.to_datetime(original_df['date_start'], errors='coerce')
        original_df = original_df.dropna(subset=['date'])

        numeric_fields = [
            'spend', 'impressions', 'clicks', 'purchases', 'purchase_value',
            'conversion_value', 'conversions', 'cpc', 'ctr'
        ]
        for col in numeric_fields:
            if col in original_df.columns:
                original_df[col] = pd.to_numeric(original_df[col], errors='coerce').fillna(0)

        # Calculate aggregated metrics per day
        grouped_df = original_df.groupby('date').agg({
            'spend': 'sum',
            'purchases': 'sum',
            'purchase_value': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'cpc': 'mean',
            'ctr': 'mean'
        }).reset_index()

        grouped_df['roas'] = grouped_df['purchase_value'] / grouped_df['spend'].replace(0, 1)
        grouped_df['cpa'] = grouped_df['spend'] / grouped_df['purchases'].replace(0, 1)
        grouped_df['click_to_conversion'] = grouped_df['purchases'] / grouped_df['clicks'].replace(0, 1)

        # Pad with missing dates for last 30 days
        last_30_days = pd.date_range(end=pd.Timestamp.today(), periods=30)
        ad_insights_df = grouped_df.set_index('date').reindex(last_30_days).fillna(0).rename_axis('date').reset_index()

        # Detect currency
        if 'account_currency' in original_df.columns:
            valid_currencies = original_df['account_currency'].dropna().astype(str).str.upper()
            currency = valid_currencies.mode()[0] if not valid_currencies.empty else "USD"
        else:
            currency = "USD"
        currency_symbol = "₹" if currency == "INR" else "$"

        combined_data = {
            "page_insights": page_data,
            "ad_insights": ad_data
        }

        # ✅ Generate key metrics + charts
        key_metrics = generate_key_metrics_section(ad_insights_df, currency_symbol=currency_symbol)

        # ✅ LLM Sections
        print("🤖 Generating Executive Summary...")
        executive_summary = await generate_llm_content(EXECUTIVE_SUMMARY_PROMPT, combined_data)

        print("🤖 Generating Account Naming & Structure analysis...")
        account_structure = await generate_llm_content(ACCOUNT_NAMING_STRUCTURE_PROMPT, combined_data)

        print("🤖 Generating Testing Activity section...")
        testing_activity = await generate_llm_content(TESTING_ACTIVITY_PROMPT, combined_data)

        print("🤖 Generating Remarketing Activity section...")
        remarketing_activity = await generate_llm_content(REMARKETING_ACTIVITY_PROMPT, combined_data)

        print("🤖 Generating Results Setup section...")
        results_setup = await generate_llm_content(RESULTS_SETUP_PROMPT, combined_data)

        # ✅ Combine sections
        sections = [
            {"title": "EXECUTIVE SUMMARY", "content": executive_summary, "charts": []},
            {"title": "ACCOUNT NAMING & STRUCTURE", "content": account_structure, "charts": []},
            {"title": "TESTING ACTIVITY", "content": testing_activity, "charts": []},
            {"title": "REMARKETING ACTIVITY", "content": remarketing_activity, "charts": []},
            {"title": "RESULTS SETUP", "content": results_setup, "charts": []},
            key_metrics
        ]

        print("📄 Generating PDF report...")
        pdf_response = generate_pdf_report(
            sections,
            ad_insights_df=ad_insights_df,
            full_ad_insights_df=original_df,
            currency_symbol=currency_symbol
        )

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
