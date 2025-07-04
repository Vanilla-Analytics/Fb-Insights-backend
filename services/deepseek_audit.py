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
from datetime import datetime, timedelta , timezone
import json

DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

def generate_chart_1(ad_insights_df):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(20, 8), dpi=200, constrained_layout=True)

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
    ax1.set_ylabel("Purchases", color="#4CAF50", fontsize=10)
    ax2.set_ylabel("Amount Spent", color="magenta", fontsize=10)

    ax1.tick_params(axis='y', labelcolor="#4CAF50", labelsize=10)
    ax2.tick_params(axis='y', labelcolor="magenta", labelsize=10)

    # X-axis formatting
    ax1.set_xticks(ad_insights_df["date"])
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax1.tick_params(axis='x', rotation=45, labelsize=8)

    # Grid
    ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    #fig.tight_layout()
    #fig.subplots_adjust(left=0.1, right=0.85, bottom=0.2, top=0.9)
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


def generate_key_metrics_section(ad_insights_df, currency_symbol="₹"):
    
    # if 'account_currency' not in ad_insights_df.columns:
    #     currency_symbol = "$"  # Default to USD
    # else:
    #     currency_symbol = ad_insights_df['account_currency'].mode()[0] if not ad_insights_df['account_currency'].mode().empty else "$"
    #     currency_symbol = "₹" if currency_symbol == "INR" else "$"

    if ad_insights_df.empty or len(ad_insights_df) < 2:
        print("⚠️ Not enough data to generate charts.")
        return "No data available for Key Metrics.", []
    
    # Calculate metrics properly
    total_spend = ad_insights_df['spend'].sum()
    total_purchases = ad_insights_df['purchases'].sum()
    total_purchase_value = ad_insights_df['purchase_value'].sum()
    total_impressions = ad_insights_df['impressions'].sum()
    total_clicks = ad_insights_df['clicks'].sum()

    metrics_summary = {
        "Amount Spent": f"{currency_symbol}{total_spend:,.2f}",
        "Purchases": int(total_purchases),
        "Purchase Value": f"{currency_symbol}{total_purchase_value:,.2f}",
        "ROAS": round(total_purchase_value / total_spend, 2) if total_spend > 0 else 0,
        "CPA": f"{currency_symbol}{round(total_spend / total_purchases, 2) if total_purchases > 0 else 0:.2f}",
        "Cost/Result": f"{currency_symbol}{round(total_spend / total_purchases, 2) if total_purchases > 0 else 0:.2f}",
        "Impressions": int(total_impressions),
        "CPM": f"{currency_symbol}{round((total_spend / total_impressions * 1000), 2) if total_impressions > 0 else 0:.2f}",
        "Link Clicks": int(total_clicks),
        "Link CPC": f"{currency_symbol}{round(total_spend / total_clicks, 2) if total_clicks > 0 else 0:.2f}",
        "CTR (link)": f"{round((total_clicks / total_impressions), 4) if total_impressions > 0 else 0:.2%}"
    }

    # metrics_summary = {
    #     "Amount Spent": f"{currency_symbol}{ad_insights_df['spend'].sum():,.2f}",
    #     "Purchases": int(ad_insights_df['purchases'].sum()),
    #     "Purchase Value": f"{currency_symbol}{ad_insights_df['purchase_value'].sum():,.2f}",
    #     "ROAS": round(ad_insights_df['roas'].mean(), 2),
    #     "CPA": f"{currency_symbol}{ad_insights_df['cpa'].mean():.2f}",
    #     "Cost/Result": f"{currency_symbol}{ad_insights_df['cpa'].mean():.2f}",
    #     "Impressions": int(ad_insights_df['impressions'].sum()),
    #     "CPM": f"{currency_symbol}{(ad_insights_df['spend'].sum() / ad_insights_df['impressions'].sum() * 1000):.2f}",
    #     "Link Clicks": int(ad_insights_df['clicks'].sum()),
    #     "Link CPC": f"{currency_symbol}{ad_insights_df['cpc'].mean():.2f}",
    #     "CTR (link)": f"{ad_insights_df['ctr'].mean():.2%}"
    # }
    

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
    fig2, ax3 = plt.subplots(figsize=(13, 5))
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
    fig3, ax5 = plt.subplots(figsize=(13, 5))
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
    fig4, ax7 = plt.subplots(figsize=(13, 5))
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

# def generate_campaign_split_charts(df):
#     import matplotlib.pyplot as plt

#     # Group by campaign
#     grouped = df[df['campaign_name'].notna()].copy()
#     grouped['spend'] = pd.to_numeric(grouped['spend'], errors='coerce').fillna(0)
#     grouped['purchase_value'] = pd.to_numeric(grouped['purchase_value'], errors='coerce').fillna(0)

#     spend_split = grouped.groupby('campaign_name')['spend'].sum().sort_values(ascending=False)
#     revenue_split = grouped.groupby('campaign_name')['purchase_value'].sum().sort_values(ascending=False)
#     roas_split = revenue_split / spend_split.replace(0, 1)

#     top_spend = spend_split.head(8)
#     top_revenue = revenue_split.head(8)
#     top_roas = roas_split.dropna().sort_values(ascending=False).head(10)

#     figs = []

#     # 1. Cost Split (Donut)
#     fig1, ax1 = plt.subplots(figsize=(3.5, 3.5))
#     wedges, texts, autotexts = ax1.pie(top_spend, labels=top_spend.index, autopct='%1.1f%%', startangle=90)
#     centre_circle = plt.Circle((0, 0), 0.70, fc='white')
#     fig1.gca().add_artist(centre_circle)
#     ax1.set_title('Cost Split', fontsize=14)
#     figs.append(("Cost Split", generate_chart_image(fig1)))

#     # 2. Revenue Split (Donut)
#     fig2, ax2 = plt.subplots(figsize=(3.5, 3.5))
#     wedges2, texts2, autotexts2 = ax2.pie(top_revenue, labels=top_revenue.index, autopct='%1.1f%%', startangle=90)
#     centre_circle = plt.Circle((0, 0), 0.70, fc='white')
#     fig2.gca().add_artist(centre_circle)
#     ax2.set_title('Revenue Split', fontsize=14)
#     figs.append(("Revenue Split", generate_chart_image(fig2)))

#     # 3. ROAS Split (Horizontal bar)
#     fig3, ax3 = plt.subplots(figsize=(5.5, 3.5))
#     ax3.barh(top_roas.index[::-1], top_roas.values[::-1], color='#ff00aa')
#     ax3.set_title('ROAS Split', fontsize=14)
#     ax3.set_xlabel("ROAS")
#     plt.tight_layout()
#     figs.append(("ROAS Split", generate_chart_image(fig3)))

#     return figs


def generate_campaign_split_charts(df, currency_symbol=None):
    if currency_symbol is None:
        currency_symbol = "₹"  # or "$" if you prefer USD fallback

    # Group by campaign - filter out rows without campaign names first
    grouped = df[df['campaign_name'].notna()].copy()
    
    # Convert numeric columns safely
    grouped['spend'] = pd.to_numeric(grouped['spend'], errors='coerce').fillna(0)
    grouped['purchase_value'] = pd.to_numeric(grouped['purchase_value'], errors='coerce').fillna(0)

    # Check if we have any valid data
    if grouped.empty or grouped['spend'].sum() == 0:
        print("⚠️ No valid campaign data available for split charts")
        return []  # Return empty list if no data

    spend_split = grouped.groupby('campaign_name')['spend'].sum().sort_values(ascending=False)
    revenue_split = grouped.groupby('campaign_name')['purchase_value'].sum().sort_values(ascending=False)
    
    # Handle division by zero for ROAS calculation
    roas_split = revenue_split / spend_split.replace(0, 1)
    roas_split = roas_split.dropna()

    # Get top campaigns (but ensure we have data)
    top_spend = spend_split.head(8) if not spend_split.empty else pd.Series(dtype=float)
    top_revenue = revenue_split.head(8) if not revenue_split.empty else pd.Series(dtype=float)
    top_roas = roas_split.sort_values(ascending=False).head(10) if not roas_split.empty else pd.Series(dtype=float)

    figs = []

    #1. Cost Split (Donut) - only if we have data
    if not top_spend.empty:
        fig1, ax1 = plt.subplots(figsize=(3.5, 3.5))
        percentages = 100 * top_spend / top_spend.sum()
        labels = [f"{name} ({pct:.1f}%)" for name, pct in zip(top_spend.index, percentages)]
        # wedges, texts, autotexts = ax1.pie(
        #     top_spend, 
        #     labels=top_spend.index, 
        #     autopct='%1.1f%%', 
        #     startangle=90,
        #     textprops={'fontsize': 8}
        # )
        wedges, texts = ax1.pie(
            top_spend,
            labels=labels,
            startangle=90,
            textprops={'fontsize': 8}  # Smaller font size to fit
        )
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig1.gca().add_artist(centre_circle)
        ax1.set_title('Cost Split', fontsize=14)
        figs.append(("Cost Split", generate_chart_image(fig1)))
    else:
        print("⚠️ No spend data available for cost split chart")

    # 2. Revenue Split (Donut) - only if we have data
    if not top_revenue.empty:
        fig2, ax2 = plt.subplots(figsize=(3.5, 3.5))
        wedges2, texts2, autotexts2 = ax2.pie(
            top_revenue, 
            labels=top_revenue.index, 
            autopct='%1.1f%%', 
            startangle=90
        )
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig2.gca().add_artist(centre_circle)
        ax2.set_title('Revenue Split', fontsize=14)
        figs.append(("Revenue Split", generate_chart_image(fig2)))
    else:
        print("⚠️ No revenue data available for revenue split chart")
    
    # # 1. Cost Split (Donut) - only if we have data
    # if not top_spend.empty:
    #     fig1, ax1 = plt.subplots(figsize=(3.5, 3.5))

    # # Generate labels with percentage included
    #     percentages = 100 * top_spend / top_spend.sum()
    #     labels = [f"{name} ({pct:.1f}%)" for name, pct in zip(top_spend.index, percentages)]

    #     wedges, texts = ax1.pie(
    #         top_spend,
    #         labels=labels,
    #         startangle=90,
    #         textprops={'fontsize': 9}
    #     )

    # # Add donut hole
    #     centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    #     fig1.gca().add_artist(centre_circle)
    #     ax1.set_title('Cost Split', fontsize=14)

    #     figs.append(("Cost Split", generate_chart_image(fig1)))
    # else:
    #     print("⚠️ No spend data available for cost split chart")


    # 3. ROAS Split (Horizontal bar) - only if we have data
    if not top_roas.empty:
        fig3, ax3 = plt.subplots(figsize=(5.5, 3.5))
        ax3.barh(top_roas.index[::-1], top_roas.values[::-1], color='#ff00aa')
        ax3.set_title('ROAS Split', fontsize=14)
        ax3.set_xlabel("ROAS")
        plt.tight_layout()
        figs.append(("ROAS Split", generate_chart_image(fig3)))
    else:
        print("⚠️ No ROAS data available for ROAS split chart")

    return figs

def generate_cost_by_campaign_chart(df):

    df['date'] = pd.to_datetime(df['date'])
    df['spend'] = pd.to_numeric(df['spend'], errors='coerce').fillna(0)
    df['campaign_name'] = df['campaign_name'].fillna("Unknown Campaign")

    grouped = df.groupby(['campaign_name', 'date'])['spend'].sum().reset_index()
    pivot_df = grouped.pivot(index='date', columns='campaign_name', values='spend').fillna(0)

    fig, ax = plt.subplots(figsize=(15, 6), dpi=200)

    for column in pivot_df.columns:
        ax.plot(pivot_df.index, pivot_df[column], label=column, linewidth=1.5, marker='o', markersize=3)

    ax.set_title("Cost By Campaigns", fontsize=16, weight='bold')
    ax.set_ylabel("Amount spent")
    ax.set_xlabel("Day")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    return ("Cost By Campaigns", generate_chart_image(fig))



async def fetch_facebook_insights(page_id: str, page_token: str):
    """Fetch Facebook page insights"""
    try:
        base_url = f"https://graph.facebook.com/v22.0/{page_id}/insights"
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
    
async def check_account_status(account_id, token):
    url = f"https://graph.facebook.com/v22.0/{account_id}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params={
            "access_token": token,
            "fields": "id,name,account_status,disable_reason"
        })
        return resp.json()

async def fetch_ad_insights(user_token: str):
    timeout = httpx.Timeout(60.0, connect=10.0)  # ✅ increase timeout
    """Fetch Facebook ad insights using user's access token"""
    try:
        url = f"https://graph.facebook.com/v22.0/me/adaccounts"
        async with httpx.AsyncClient(timeout=timeout) as client:
            acc_resp = await client.get(url, params={
                "access_token": user_token,
                #"fields": "id,name,account_status,disable_reason,adsets{id,name}"
                "fields": "id,name,account_status,disable_reason,account_currency,adsets{id,name}"

            })
            acc_resp.raise_for_status()
            accounts = acc_resp.json().get("data", [])
            print("✅ 'account_currency' values from accounts:", [acc.get("account_currency") for acc in accounts])


            if not accounts:
                print("⚠️ No ad accounts found for this user")
                return []

            insights_data = []
            for acc in accounts:
                try:
                    print(f"🔍 Processing account: {acc.get('name')} ({acc.get('id')})")

                    # Skip accounts with no adsets
                    if 'adsets' not in acc or not acc['adsets'].get('data'):
                        print(f"⚠️ No adsets found for account {acc.get('name')}")
                        continue

                    ad_url = f"https://graph.facebook.com/v22.0/{acc['id']}/insights"
                    now = datetime.now(timezone.utc)
                    safe_until = (now - timedelta(days=1)).strftime("%Y-%m-%d")
                    safe_since = (now - timedelta(days=60)).strftime("%Y-%m-%d")


                    params = {
                        "fields": "campaign_name,adset_name,ad_name,spend,impressions,clicks,cpc,ctr,actions,action_values,date_start,account_currency,account_name",
                        "time_range": json.dumps({
                            # "since": (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%d"),
                            # "until": datetime.now(timezone.utc).strftime("%Y-%m-%d")
                            "since": safe_since,
                            "until": safe_until
                        }),
                        "time_increment": 1,
                        "level": "ad",
                        "access_token": user_token
                    }

                    detailed_response = await client.get(ad_url, params=params)
                    detailed_response.raise_for_status()

                    ad_results = []
                    data_page = detailed_response.json()
                    ad_results.extend(data_page.get("data", []))
                    print(f"🔍 Raw insight response: {detailed_response.text}")


                    while 'paging' in data_page and 'next' in data_page['paging']:
                        next_url = data_page["paging"]["next"]
                        next_response = await client.get(next_url)
                        next_response.raise_for_status()
                        data_page = next_response.json()
                        ad_results.extend(data_page.get("data", []))

                    print(f"📊 Total detailed insights: {len(ad_results)}")

                    for ad in ad_results:
                        ad["account_currency"] = acc.get("account_currency", "USD")
                        insights_data.append(ad)

                except Exception as e:
                    import traceback
                    print(f"⚠️ Error processing account {acc.get('id')}: {str(e)}")
                    traceback.print_exc()

            return insights_data

    except Exception as e:
        print(f"❌ Error in fetch_ad_insights: {str(e)}")
        return []

#-----------------------------------------------------------------------------------------------
# async def fetch_ad_insights(user_token: str):
#     """Fetch Facebook ad insights"""
#     try:
#         url = f"https://graph.facebook.com/v22.0/me/adaccounts"
#         async with httpx.AsyncClient() as client:
#             # First get accounts with more detailed fields
#             acc_resp = await client.get(url, params={
#                 "access_token":user_token,
#                 "fields": "id,name,account_status,disable_reason,adsets{id,name}"
#             })
#             acc_resp.raise_for_status()
#             accounts = acc_resp.json().get("data", [])
            
#             if not accounts:
#                 print("⚠️ No ad accounts found for this user")
#                 return []

#             insights_data = []
#             for acc in accounts:
#                 try:
#                     print(f"🔍 Processing account: {acc.get('name')} ({acc.get('id')})")
                    
#                     # Check if account has any active adsets first
#                     if 'adsets' not in acc or not acc['adsets'].get('data'):
#                         print(f"⚠️ No adsets found for account {acc.get('name')}")
#                         continue
                        
#                     # Get insights with broader time range and simpler fields first
#                     #ad_url = f"https://graph.facebook.com/v22.0/{acc['id']}/insights"
#                     ad_url = f"https://graph.facebook.com/v22.0/me/adaccounts?fields=id,name,account_status,adsets{id}&access_token=<EAAcIf6mRqdcBOZBncynonfReAZAzoJmkfkYXZAFqnZCzlhgPk823WHQ60WG6JMiMwBnYYriLP6IxJuJMn63puvANFyA69E1xm1MrGmoZAM5BkxnJKRZBZC9SdyXp8gBQLk2QnmLAtzzJR4MkHsywDCBreI61kp0OjlUhDyv73q3mcjmxahnZAFjeaMotPgyR3LWxPtFUfMnXjI1viFPTAJVlU1ylZAG546ZCLSpQZDZD>"

#                     params = {
#                         "fields": "spend,impressions,date_start",
#                         "date_preset": "maximum",  # Get all available data
#                         "time_increment": 1,
#                         "level": "ad",
#                         "access_token": user_token
#                     }
                    
#                     print(f"📡 Initial insights request for account {acc['id']}")
#                     insights_response = await client.get(ad_url, params=params)
#                     insights_response.raise_for_status()
                    
#                     data = insights_response.json().get("data", [])
#                     print(f"📊 Initial insights count: {len(data)}")
                    
#                     if not data:
#                         print(f"⚠️ No insights data for account {acc['id']}")
#                         continue
                        
#                     # If we got some data, now request detailed fields
#                     detailed_params = {
#                         "fields": "campaign_name,adset_name,ad_name,spend,impressions,clicks,cpc,ctr,actions,action_values,date_start,account_currency,account_name",
#                         "time_range": json.dumps({
#                             "since": (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%d"),
#                             "until": datetime.now(timezone.utc).strftime("%Y-%m-%d")
#                         }),
#                         "time_increment": 1,
#                         "level": "ad",
#                         "access_token": user_token
#                     }
                    
#                     detailed_response = await client.get(ad_url, params=detailed_params)
#                     detailed_response.raise_for_status()
                    
#                     ad_results = []
#                     data_page = detailed_response.json()
#                     ad_results.extend(data_page.get("data", []))
                    
#                     # Handle pagination
#                     while 'paging' in data_page and 'next' in data_page['paging']:
#                         next_url = data_page["paging"]["next"]
#                         data_page = await client.get(next_url)
#                         data_page = data_page.json()
#                         ad_results.extend(data_page.get("data", []))
                    
#                     print(f"📊 Total detailed insights: {len(ad_results)}")
                    
#                     for ad in ad_results:
#                         ad["account_currency"] = "USD"  # Default, adjust as needed
#                         insights_data.append(ad)
                        
#                 except Exception as e:
#                     print(f"⚠️ Error processing account {acc.get('id')}: {str(e)}")
#                     continue
            
#             return insights_data
            
#     except Exception as e:
#         print(f"❌ Error in fetch_ad_insights: {str(e)}")
#         return []

#-----------------------------------------------------------------------------------------------


import json
import httpx

MAX_TOKEN_LIMIT = 8000  # Maximum length of JSON string to send
MAX_AD_ITEMS = 30       # Max ad entries to keep for DeepSeek

def truncate_ad_data(data: dict, max_items: int = MAX_AD_ITEMS) -> dict:
    """Return a smaller copy of ad_insights for LLM prompt"""
    truncated = data.copy()
    if "ad_insights" in truncated and isinstance(truncated["ad_insights"], list):
        truncated["ad_insights"] = truncated["ad_insights"][:max_items]
    return truncated

async def generate_llm_content(prompt: str, data: dict) -> str:
    """Generate content using DeepSeek LLM with safe truncation"""
    try:
        # 🔹 Step 1: Truncate long ad_insights
        truncated_data = truncate_ad_data(data)

        # 🔹 Step 2: Convert to JSON string
        try:
            data_str = json.dumps(truncated_data, indent=2)
        except Exception as e:
            print(f"⚠️ Error serializing data to JSON: {e}")
            data_str = str(truncated_data)  # fallback

        # 🔹 Step 3: Truncate further if still too large
        if len(data_str) > MAX_TOKEN_LIMIT:
            print(f"⚠️ Even truncated data too large ({len(data_str)} chars). Truncating...")
            data_str = data_str[:MAX_TOKEN_LIMIT] + "\n\n...[truncated]"

        # 🔹 Step 4: Prepare prompt
        user_prompt = f"{prompt}\n\nAnalyze the following Facebook data:\n{data_str}"

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a Meta Ads expert who writes audit reports."},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7
        }

        # 🔹 Step 5: Make API call
        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(
                DEEPSEEK_API_URL,
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload
            )
            res.raise_for_status()
            response_data = res.json()
            return response_data["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"❌ Error generating LLM content: {str(e)}")
        return "⚠️ Unable to generate this section due to large dataset or API error."



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
        ad_raw = []        
        
        if not ad_data:
            print("🚨 Raw ad_data returned from Facebook:")
            ad_raw = await fetch_ad_insights(user_token)
            print(ad_raw[:2])

        if not ad_data:
            raise ValueError("❌ All ad insights entries are missing 'date_start' — cannot proceed.")
        
        PURCHASE_KEYS = [
            "offsite_conversion.purchase",
            "offsite_conversion.fb_pixel_purchase",
            "offsite_conversion.fb_pixel_custom",
            "offsite_conversion.custom.1408006162945363",
            "offsite_conversion.custom.587738624322885"
        ]

        for ad in ad_data:
            try:
                actions = {d["action_type"]: float(d["value"]) for d in ad.get("actions", []) if "action_type" in d and "value" in d}
                values = {d["action_type"]: float(d["value"]) for d in ad.get("action_values", []) if "action_type" in d and "value" in d}
            except Exception as e:
                print(f"⚠️ Error parsing actions in ad: {e}")
                actions, values = {}, {}

            ad["purchases"] = sum(actions.get(k, 0) for k in PURCHASE_KEYS)
            ad["purchase_value"] = sum(values.get(k, 0) for k in PURCHASE_KEYS)
            ad["link_clicks"] = actions.get("link_click", 0)




        # Create original DataFrame with date_start intact
        original_df = pd.DataFrame(ad_data)
        # 🚨 Check if account_currency is missing
        if 'account_currency' not in original_df.columns:
            print("⚠️ 'account_currency' column missing in original_df.")
        else:
            print("✅ 'account_currency' found:", original_df['account_currency'].unique())
        for col in ['spend', 'impressions', 'clicks', 'purchases', 'purchase_value', 'cpc', 'ctr', 'link_clicks']:
            original_df[col] = pd.to_numeric(original_df.get(col, 0), errors='coerce').fillna(0)

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
        #last_30_days = pd.date_range(end=pd.Timestamp.today(), periods=30)
        #ad_insights_df = grouped_df.set_index('date').reindex(last_30_days).fillna(0).rename_axis('date').reset_index()

        #cutoff = pd.Timestamp.today() - pd.Timedelta(days=30)
        #ad_insights_df = grouped_df[grouped_df['date'] >= cutoff].copy()

        cutoff = pd.Timestamp.today() - pd.Timedelta(days=60)
        ad_insights_df = grouped_df[grouped_df['date'] >= cutoff].copy()

        if ad_insights_df.empty:
            print("⚠️ No data in last 30 days. Using last available 30 records.")
            ad_insights_df = grouped_df.tail(30)



        # Detect currency-----------------------
        # if 'account_currency' in original_df.columns:
        #     valid_currencies = original_df['account_currency'].dropna().astype(str).str.upper()
        #     currency = valid_currencies.mode()[0] if not valid_currencies.empty else "USD"
        # else:
        #     currency = "USD"
        # currency_symbol = "₹" if currency == "INR" else "$"

        # Detect currency more reliably
        currency = "USD"  # Default
        currency_symbol = "$"

        # if 'account_currency' in original_df.columns:
        # # Get the most frequent non-null currency
        #     valid_currencies = original_df['account_currency'].dropna().astype(str).str.upper()
        #     if not valid_currencies.empty:
        #     # Check for any INR occurrences first
        #         if "INR" in valid_currencies.values:
        #             currency = "INR"
        #             currency_symbol = "₹"
        #         else:
        #         # Fall back to mode if no INR found
        #             currency = valid_currencies.mode()[0]
        #             currency_symbol = "₹" if currency == "INR" else "$"

        def detect_currency(df):
            if 'account_currency' not in df.columns:
                print("⚠️ No 'account_currency' column found in the DataFrame")
                return "USD", "$"
    
            # Simple mapping of currency codes to symbols
            currency_symbols = {
                "INR": "₹",  # Indian Rupee
                "USD": "$",  # US Dollar
                # Add more currencies as needed
            }
            
            # Get most frequent currency from the data
            currencies = df['account_currency'].dropna().astype(str).str.strip().str.upper()
            if currencies.empty:
                print("⚠️ No valid currency values found in 'account_currency' column")
                return "USD", "$"
            
            # Print unique currencies for debugging
            unique_currencies = currencies.unique()
            print(f"🔍 Unique currency values found: {unique_currencies}")
            
            # Get the most frequent currency
            currency = currencies.mode()[0] if not currencies.mode().empty else "USD"
            
            # Get the symbol for this currency (default to $ if not in our mapping)
            currency_symbol = currency_symbols.get(currency, "$")
            
            print(f"✅ Using currency: {currency} with symbol: {currency_symbol}")
            return currency, currency_symbol

        currency, currency_symbol = detect_currency(original_df)

        print(f"💰 Detected account currency: {currency} → Using symbol: {currency_symbol}")


        combined_data = {
            "page_insights": page_data,
            "ad_insights": ad_data
        }

        # ✅ Generate key metrics + charts
        key_metrics = generate_key_metrics_section(ad_insights_df, currency_symbol=currency_symbol)
        split_charts = generate_campaign_split_charts(original_df, currency_symbol)
        


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
        cost_by_campaign_chart = generate_cost_by_campaign_chart(original_df)
        sections.append({
            "title": "COST BY CAMPAIGNS",
            "content": "",  # No paragraph needed
            "charts": [cost_by_campaign_chart],
            "contains_table": False
        })
        sections.append({
            "title": "Daily Campaign Performance Summary",
            "content": "",
            "charts": [],
            "contains_table": True  # ✅ mark table-only pages like this
        })
        sections.append({
            "title": "CAMPAIGN PERFORMANCE SUMMARY",
            "content": "",
            "charts": [],
            "contains_table": True  # ✅ mark table-only pages like this
        })

        print("📄 Generating PDF report...")
        pdf_response = generate_pdf_report(
            sections,
            ad_insights_df=ad_insights_df,
            full_ad_insights_df=original_df,
            currency_symbol=currency_symbol,
            split_charts=split_charts
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
