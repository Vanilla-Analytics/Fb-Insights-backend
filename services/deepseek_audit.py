# services/deepseek_audit.py
import httpx
import os
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

DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

def generate_chart_1(ad_insights_df):
    
    fig, ax1 = plt.subplots(figsize=(16, 6))  # ✅ Wider chart

    # Fill missing values
    ad_insights_df['purchase_value'] = ad_insights_df['purchase_value'].fillna(0)
    ad_insights_df['spend'] = ad_insights_df['spend'].fillna(0)

    # ✅ Green bars for Purchase Conversion Value
    ax1.bar(
        ad_insights_df["date"],
        ad_insights_df["purchase_value"],
        color="#4CAF50",  # ✅ darker green for visibility
        edgecolor="black",  # ✅ optional border for better contrast
        linewidth=0.5,
        label="Purchase Conversion Value"
    )
    ax1.set_ylabel("Purchase Conversion Value", color="#6B8E23", fontsize=12)
    ax1.tick_params(axis='y', labelcolor="#6B8E23", labelsize=10)
    ax1.spines['left'].set_linewidth(1.5)

    # ✅ Thicker magenta line for Amount Spent
    ax2 = ax1.twinx()
    ax2.plot(
        ad_insights_df["date"],
        ad_insights_df["spend"],
        color="magenta",
        marker="o",
        label="Amount Spent",
        linewidth=2.5
    )
    ax2.set_ylabel("Amount Spent", color="magenta", fontsize=12)
    ax2.tick_params(axis='y', labelcolor="magenta", labelsize=10)
    ax2.spines['right'].set_linewidth(1.5)

    # Axis limits
    ax1.set_ylim(0, ad_insights_df['purchase_value'].max() * 1.2)
    ax2.set_ylim(0, ad_insights_df['spend'].max() * 1.2)

    # ✅ Format X-axis
    ax1.tick_params(axis='x', labelrotation=45, labelsize=10)
    #ax1.xaxis.set_major_formatter(mticker.DateFormatter('%d-%b'))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))  # ✅ Correct usage


    # Grid and layout
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
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


def generate_key_metrics_section(ad_insights_df):
    if ad_insights_df.empty or len(ad_insights_df) < 2:
        print("⚠️ Not enough data to generate charts.")
        return "No data available for Key Metrics.", []

    metrics_summary = {
        "Amount Spent": f"₹{ad_insights_df['spend'].sum():,.2f}",
        "Purchases": int(ad_insights_df['purchases'].sum()),
        "Purchase Value": f"₹{ad_insights_df['purchase_value'].sum():,.2f}",
        "ROAS": round(ad_insights_df['roas'].mean(), 2),
        "CPA": f"₹{ad_insights_df['cpa'].mean():.2f}",
        "Cost/Result": f"₹{ad_insights_df['cpa'].mean():.2f}",
        "Impressions": int(ad_insights_df['impressions'].sum()),
        "CPM": f"₹{(ad_insights_df['spend'].sum() / ad_insights_df['impressions'].sum() * 1000):.2f}",
        "Link Clicks": int(ad_insights_df['clicks'].sum()),
        "Link CPC": f"₹{ad_insights_df['cpc'].mean():.2f}",
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
        url = f"https://graph.facebook.com/v18.0/me/adaccounts"
        async with httpx.AsyncClient() as client:
            acc_resp = await client.get(url, params={"access_token": page_token})
            acc_resp.raise_for_status()
            if acc_resp.status_code != 200:
                print("⚠️ Facebook API Error:", acc_resp.text)
                acc_resp.raise_for_status()

            accounts = acc_resp.json().get("data", [])
            print("📡 Ad Accounts fetched:", accounts)
            from datetime import datetime, timedelta

            since = (datetime.today() - timedelta(days=60)).strftime('%Y-%m-%d')
            until = datetime.today().strftime('%Y-%m-%d')
            insights_data = []
            for acc in accounts:
                try:
                    ad_url = f"https://graph.facebook.com/v18.0/{acc['id']}/insights"
                    ad_params = {
                        "fields": "campaign_name,adset_name,ad_name,spend,impressions,clicks,cpc,ctr",
                        "time_range": {"since": since, "until": until},
                        #"date_preset": "last_60_days",
                        #"date_preset":"maximum",
                        "time_increment": 1,  # 👈 daily breakdown
                        "level": "ad",        # 👈 required to enable daily granularity
                        "access_token": page_token
                    }
                    insights_resp = await client.get(ad_url, params=ad_params)
                    if insights_resp.status_code == 200:
                        insights_data.extend(insights_resp.json().get("data", []))
                        print(f"📊 Insights from account {acc['id']}:", insights_resp.json().get("data", []))

                    else:
                        print(f"⚠️ Warning: Failed to fetch insights for account {acc['id']}")
                        print(f"🔍 Response status: {insights_resp.status_code}")
                        print(f"🔍 Response content: {insights_resp.text}")
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


async def generate_audit(page_id: str,user_token: str, page_token: str):
    """Generate audit report and return PDF"""
    try:
        print("🔄 Starting audit generation...")
        
        # Validate environment variables
        if not DEEPSEEK_API_URL or not DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_URL and DEEPSEEK_API_KEY environment variables must be set")
        
        # Fetch data from Facebook
        print("📊 Fetching Facebook data...")
        page_data = await fetch_facebook_insights(page_id, page_token)
        ad_data = await fetch_ad_insights(user_token)
        # Sanitize and ensure 'date_start' exists
        ad_data = [d for d in ad_data if isinstance(d, dict) and 'date_start' in d and d.get('date_start')]
        if not ad_data:
            raise ValueError("❌ All ad insights entries are missing 'date_start' — cannot proceed.")

        # expected_keys = ['date_start', 'spend', 'impressions', 'clicks', 'cpc', 'ctr']
        # ad_data = [{k: d.get(k, None) for k in expected_keys} for d in ad_data if isinstance(d, dict)]

        # if not ad_data:
        #     raise ValueError("❌ No ad insights returned from Facebook. Cannot generate report.")

        ad_insights_df = pd.DataFrame(ad_data)
        # Ensure 'date' is present for charting and grouping
        if 'date_start' in ad_insights_df.columns:
            ad_insights_df['date'] = pd.to_datetime(ad_insights_df['date_start'], errors='coerce')
        else:
            raise ValueError("❌ Cannot parse 'date' because 'date_start' is missing.")


        # Ensure all required columns exist, even if filled with zeros
        #expected_cols = ['date', 'spend', 'purchase_value', 'purchases', 'cpa', 'impressions','ctr', 'clicks', 'click_to_conversion', 'roas', 'cpc']
        expected_cols = ['spend', 'purchase_value', 'purchases', 'cpa', 'impressions','ctr', 'clicks', 'click_to_conversion', 'roas', 'cpc']


        for col in expected_cols:
            if col not in ad_insights_df.columns:
                ad_insights_df[col] = 0


        combined_data = {
            "page_insights": page_data,
            "ad_insights": ad_data
        }
       
        print("🔎 Columns in ad_insights_df:", ad_insights_df.columns.tolist())

        # Ensure fallback/derived fields exist
        if 'purchase_value' not in ad_insights_df.columns:
            ad_insights_df['purchase_value'] = ad_insights_df.get('conversion_value', 0)

        if 'purchases' not in ad_insights_df.columns:
            ad_insights_df['purchases'] = ad_insights_df.get('conversions', 0)

        if 'clicks' not in ad_insights_df.columns:
            ad_insights_df['clicks'] = 1

        if 'spend' not in ad_insights_df.columns:
            ad_insights_df['spend'] = 1

# ✅ Convert all key fields to numeric
        numeric_fields = [
            'spend', 'impressions', 'clicks', 'purchases', 'purchase_value','conversion_value', 'conversions', 'cpc', 'ctr'
        ]

        for col in numeric_fields:
            if col in ad_insights_df.columns:
                ad_insights_df[col] = pd.to_numeric(ad_insights_df[col], errors='coerce').fillna(0)
            
        if 'date_start' in ad_insights_df.columns:
            print("📋 Checking date_start content:", ad_insights_df['date_start'].head(3))
        else:
            print("⚠️ 'date_start' column is missing entirely.")

#----------------------------------------------------------------------------------------
          # ✅ Make sure 'date_start' exists and convert to datetime
            if 'date_start' not in ad_insights_df.columns:
                raise ValueError("❌ 'date_start' column is missing in ad insights data.")

            ad_insights_df = ad_insights_df.dropna(subset=['date_start'])
            if ad_insights_df.empty:
                raise ValueError("❌ All 'date_start' values are empty or invalid.")

            # Convert and filter by date
            ad_insights_df['date_start'] = ad_insights_df['date_start'].astype(str)
            ad_insights_df['date'] = pd.to_datetime(ad_insights_df['date_start'], errors='coerce')
            ad_insights_df = ad_insights_df.dropna(subset=['date'])
            print("🧪 Dates after parsing:", ad_insights_df['date'].dropna().unique())


            # cutoff_date = pd.Timestamp.today() - pd.Timedelta(days=60)
            # ad_insights_df = ad_insights_df[ad_insights_df['date'] >= cutoff_date]
            ad_insights_df = ad_insights_df.sort_values('date', ascending=False)

            # ✅ Convert to numeric to avoid aggregation issues
            cols_to_numeric = ['spend', 'purchases', 'purchase_value', 'cpa', 'impressions', 'clicks', 'cpc', 'ctr']
            for col in cols_to_numeric:
                if col in ad_insights_df.columns:
                    ad_insights_df[col] = pd.to_numeric(ad_insights_df[col], errors='coerce').fillna(0)

            # ✅ Group by date (one row per day)
            grouped_df = ad_insights_df.groupby('date').agg({
                'spend': 'sum',
                'purchases': 'sum',
                'purchase_value': 'sum',
                'impressions': 'sum',
                'clicks': 'sum',
                'cpc': 'mean',
                'ctr': 'mean'
            }).reset_index()

            
            # ad_insights_df = ad_insights_df.sort_values("date", ascending=False)
            # ad_insights_df = ad_insights_df.drop_duplicates(subset="date", keep="first")  # keep only one row per date
            # ad_insights_df = ad_insights_df.head(60).sort_values("date")  # oldest to newest for display
            # grouped_df['roas'] = grouped_df['purchase_value'] / grouped_df['spend'].replace(0, 1)
            # grouped_df['cpa'] = grouped_df['spend'] / grouped_df['purchases'].replace(0, 1)
            # grouped_df['click_to_conversion'] = grouped_df['purchases'] / grouped_df['clicks'].replace(0, 1)
            # pdf_response = generate_pdf_report(sections, ad_insights_df=ad_insights_df)

            # ad_insights_df = grouped_df
            # print("📆 Final grouped dates:", ad_insights_df['date'].dt.strftime("%Y-%m-%d").tolist())

            # ✅ Group by date (ensure one row per day)
            grouped_df = ad_insights_df.groupby('date').agg({
                'spend': 'sum',
                'purchases': 'sum',
                'purchase_value': 'sum',
                'impressions': 'sum',
                'clicks': 'sum',
                'cpc': 'mean',
                'ctr': 'mean'
            }).reset_index()

            # ✅ Add calculated fields
            grouped_df['roas'] = grouped_df['purchase_value'] / grouped_df['spend'].replace(0, 1)
            grouped_df['cpa'] = grouped_df['spend'] / grouped_df['purchases'].replace(0, 1)
            grouped_df['click_to_conversion'] = grouped_df['purchases'] / grouped_df['clicks'].replace(0, 1)

            # ✅ Keep only last 60 unique days
            ad_insights_df = grouped_df.sort_values('date', ascending=False).head(60).sort_values('date')

            # ✅ Feed final DataFrame to PDF
            print("📆 Final grouped dates:", ad_insights_df['date'].dt.strftime("%Y-%m-%d").tolist())
            #pdf_response = generate_pdf_report(sections, ad_insights_df=ad_insights_df)



        #----------------------------------------------------------------------------------------------------
        #key_metrics = generate_key_metrics_section(ad_insights_df)
        # if not ad_insights_df.empty:
        #     key_metrics = generate_key_metrics_section(ad_insights_df)
        # else:
        #     key_metrics = {
        #     "title": "KEY METRICS",
        #     "content": "No ad data available to generate Key Metrics.",
        #     "charts": []
        # }

        # ✅ Group and aggregate
        grouped_df = ad_insights_df.groupby('date').agg({
            'spend': 'sum',
            'purchases': 'sum',
            'purchase_value': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'cpc': 'mean',
            'ctr': 'mean'
        }).reset_index()

        # ✅ Add calculated fields
        grouped_df['roas'] = grouped_df['purchase_value'] / grouped_df['spend'].replace(0, 1)
        grouped_df['cpa'] = grouped_df['spend'] / grouped_df['purchases'].replace(0, 1)
        grouped_df['click_to_conversion'] = grouped_df['purchases'] / grouped_df['clicks'].replace(0, 1)

        # ✅ Filter last 60 days (unique)
        ad_insights_df = grouped_df.sort_values('date', ascending=False).head(60).sort_values('date')

        # ✅ NOW generate key metrics with grouped data
        key_metrics = generate_key_metrics_section(ad_insights_df)


        # Generate Executive Summary
        print("🤖 Generating Executive Summary...")
        executive_summary = await generate_llm_content(EXECUTIVE_SUMMARY_PROMPT, combined_data)
        print("✅ Executive Summary generated successfully")

        # Generate Account Naming & Structure analysis
        print("🤖 Generating Account Naming & Structure analysis...")
        account_structure = await generate_llm_content(ACCOUNT_NAMING_STRUCTURE_PROMPT, combined_data)
        print("✅ Account Naming & Structure analysis generated successfully")

        print("🤖 Generating Testing Activity section...")
        testing_activity = await generate_llm_content(TESTING_ACTIVITY_PROMPT, combined_data)
        print("✅ Testing Activity generated successfully")

        print("🤖 Generating Remarketing Activity section...")
        remarketing_activity = await generate_llm_content(REMARKETING_ACTIVITY_PROMPT, combined_data)
        print("✅ Remarketing Activity generated successfully")

        print("🤖 Generating Results Setup section...")
        results_setup = await generate_llm_content(RESULTS_SETUP_PROMPT, combined_data)
        print("✅ Results Setup generated successfully")

        # Prepare sections for PDF
        sections = [
            {
                "title": "EXECUTIVE SUMMARY",
                "content": executive_summary,
                "charts": []
            },
            {
                "title": "ACCOUNT NAMING & STRUCTURE",
                "content": account_structure,
                "charts": []
            },
            {
                "title": "TESTING ACTIVITY",
                "content": testing_activity,
                "charts": []
            },
            {
                "title": "REMARKETING ACTIVITY",
                "content": remarketing_activity
            },
            {
                "title": "RESULTS SETUP",
                "content": results_setup,
                "charts": []
            },
            key_metrics
        ]

        # Generate PDF
        print("✅ Final PDF table date count:", len(ad_insights_df), ad_insights_df['date'].dt.strftime("%Y-%m-%d").tolist())

        print("📄 Generating PDF report...")
        pdf_response = generate_pdf_report(sections, ad_insights_df=ad_insights_df)
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