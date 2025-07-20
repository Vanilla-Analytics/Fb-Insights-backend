# services/deepseek_audit.py
# services/deepseek_audit.py
import httpx
import os
import requests
from datetime import datetime, timedelta
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from services.prompts import EXECUTIVE_SUMMARY_PROMPT, ACCOUNT_NAMING_STRUCTURE_PROMPT
from services.prompts import TESTING_ACTIVITY_PROMPT
from services.prompts import REMARKETING_ACTIVITY_PROMPT
from services.prompts import RESULTS_SETUP_PROMPT
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
#from services.generate_pdf import generate_pdf_report
from datetime import datetime, timedelta , timezone
import json


DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

def generate_chart_1(ad_insights_df):
   
  

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

def generate_key_metrics_section(ad_insights_df, currency_symbol="₹"):

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


    summary_text = "\n".join([f"{k}: {v}" for k, v in metrics_summary.items()])

    # Charts
    chart_imgs = []


    # Chart 1: Amount Spent vs Purchase Conversion Value
    
    fig1 = generate_chart_1(ad_insights_df)
    chart_imgs.append(("# ", generate_chart_image(fig1)))
   
   # Chart 2: Purchases vs ROAS    
   
    purchases_df = ad_insights_df.sort_values("date")[-30:]

    fig2, ax3 = plt.subplots(figsize=(13, 5))
    bar_width = 0.8

    ax3.bar(purchases_df["date"], purchases_df["purchases"], width=bar_width, color="#0d0c42", label="Purchases")
    ax3.set_ylabel("Purchases", color="#0d0c42")
    ax3.tick_params(axis='y', labelcolor="#0d0c42")
    ax3.set_ylim(0, purchases_df["purchases"].max() * 1.2)

    # Format x-axis
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.xticks(rotation=45)

    ax4 = ax3.twinx()
    ax4.plot(purchases_df["date"], purchases_df["roas"], color="#ff00aa", marker="o", label="ROAS")
    ax4.set_ylabel("ROAS", color="#ff00aa")
    ax4.tick_params(axis='y', labelcolor="#ff00aa")
    ax4.set_ylim(0, purchases_df["roas"].max() * 1.2)

    ax3.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.title("Purchases vs ROAS", fontsize=14)

    plt.tight_layout()
    chart_imgs.append(("Purchases vs ROAS", generate_chart_image(fig2)))

   
   # Chart 3: CPA vs Link CPC
   
    cpa_df = ad_insights_df.sort_values("date")[-30:]

    fig3, ax5 = plt.subplots(figsize=(13, 5))
    bar_width = 0.8

    ax5.bar(cpa_df["date"], cpa_df["cpa"], width=bar_width, color="blue", label="CPA")
    ax5.set_ylabel("CPA", color="blue")
    ax5.tick_params(axis='y', labelcolor="blue")
    ax5.set_ylim(0, cpa_df["cpa"].max() * 1.2)

    ax5.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.xticks(rotation=45)

    ax6 = ax5.twinx()
    ax6.plot(cpa_df["date"], cpa_df["cpc"], color="pink", marker="o", label="Link CPC")
    ax6.set_ylabel("Link CPC", color="pink")
    ax6.tick_params(axis='y', labelcolor="pink")
    ax6.set_ylim(0, cpa_df["cpc"].max() * 1.2)

    ax5.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.title("CPA vs Link CPC", fontsize=14)

    plt.tight_layout()
    chart_imgs.append(("CPA vs Link CPC", generate_chart_image(fig3)))
      
# Chart 4: Click to Conversion vs CTR
    click_df = ad_insights_df.sort_values("date")[-30:]

    fig4, ax7 = plt.subplots(figsize=(13, 5))
    bar_width = 0.8

    ax7.bar(click_df["date"], click_df["click_to_conversion"], width=bar_width, color="pink", label="Click to Conversion")
    ax7.set_ylabel("Click to Conversion", color="pink")
    ax7.tick_params(axis='y', labelcolor="pink")
    ax7.set_ylim(0, click_df["click_to_conversion"].max() * 1.2)

    ax7.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax7.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.xticks(rotation=45)

    ax8 = ax7.twinx()
    ax8.plot(click_df["date"], click_df["ctr"], color="darkblue", marker="o", label="CTR")
    ax8.set_ylabel("CTR", color="darkblue")
    ax8.tick_params(axis='y', labelcolor="darkblue")
    ax8.set_ylim(0, click_df["ctr"].max() * 1.2)

    ax7.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.title("Click to Conversion vs CTR", fontsize=14)

    plt.tight_layout()
    chart_imgs.append(("Click to Conversion vs CTR", generate_chart_image(fig4)))


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

async def generate_ad_fatigue_summary(full_df: pd.DataFrame, currency_symbol: str) -> str:
    from services.deepseek_audit import generate_llm_content

    if full_df.empty or 'ad_name' not in full_df.columns:
        return "⚠️ No ad-level data available to summarize fatigue."

    df = full_df.copy()
    df['impressions'] = pd.to_numeric(df['impressions'], errors='coerce').fillna(0)
    #df['reach'] = pd.to_numeric(df.get('reach', df['impressions']), errors='coerce').fillna(1)
    #df['reach'] = pd.to_numeric(df['reach'], errors='coerce').fillna(1)
    # Fallback: if 'reach' column is missing, use 'impressions'
    if 'reach' not in df.columns:
        print("⚠️ 'reach' column missing, using 'impressions' as fallback.")
        df['reach'] = df['impressions']

    df['reach'] = pd.to_numeric(df['reach'], errors='coerce').fillna(1)

    df['spend'] = pd.to_numeric(df['spend'], errors='coerce').fillna(0)
    df['purchase_value'] = pd.to_numeric(df['purchase_value'], errors='coerce').fillna(0)
    df['purchases'] = pd.to_numeric(df['purchases'], errors='coerce').fillna(0)

    df['frequency'] = df['impressions'] / df['reach'].replace(0, 1)
    df['roas'] = df['purchase_value'] / df['spend'].replace(0, 1)
    df['cpa'] = df['spend'] / df['purchases'].replace(0, 1)
    df['ctr'] = pd.to_numeric(df['ctr'], errors='coerce').fillna(0)

    records = df[['ad_name', 'frequency', 'ctr', 'roas', 'cpa']].to_dict(orient='records')
    summary_data = {"ads": records}

    prompt = (
        f"Based on the following Meta Ads data, write a short professional paragraph (max 4–5 lines) summarizing signs of ad fatigue.\n\n"
        f"Include:\n"
        f"1. Whether frequency is too high (>3.5) for some ads with poor CTR or ROAS.\n"
        f"2. Whether stable frequency ads are performing better.\n"
        f"3. Trends in CPM if available.\n"
        f"4. 1 actionable recommendation.\n"
        f"Use {currency_symbol} in values where needed.\n"
        f"Do not list all ads — just highlight general observations with insight."
    )

    return await generate_llm_content(prompt, summary_data)

def build_demographic_summary_prompt(demographic_df, currency_symbol):
    """
    Builds a prompt for LLM based on demographic data.
    """
    table_preview = demographic_df.to_string(index=False)
    prompt = (
        f"You are a marketing analytics expert.\n"
        f"Analyze the following demographic data:\n\n"
        f"{table_preview}\n\n"
        f"Write a concise executive summary (~4-5 lines) highlighting:\n"
        f"1. Best performing Age/Gender groups (high ROAS, low CPA)\n"
        f"2. Underperforming segments (low ROAS or high CPA)\n"
        f"3. Strategic recommendations to focus ad budget efficiently.\n\n"
        f"Use {currency_symbol} for monetary values. Keep it professional."
    )
    return prompt
 


def draw_donut_chart(values, labels, title):    
    if values.sum() <= 0 or not np.all(np.isfinite(values)):
        raise ValueError("Invalid values for donut chart.")
    # Truncate labels to 4 words
    def truncate_label(label, max_words=4):
        tokens = label.split()
        return " ".join(tokens[:max_words]) + "..." if len(tokens) > max_words else label
    truncated_labels = [truncate_label(label) for label in labels]
    percentages = 100 * values / values.sum()
    color_map = plt.get_cmap('tab20c')
    colors = color_map.colors[:len(truncated_labels)]
    fig, ax = plt.subplots(figsize=(5, 5))  
    # Pie chart with no labels
    wedges, _ = ax.pie(
        values,
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.4)
    )
    ax.text(0, 0, "100%", ha='center', va='center', fontsize=14, weight='bold')
    # Add legend to the right
    ax.legend(
        wedges,
        [f"{label} ({pct:.1f}%)" for label, pct in zip(truncated_labels, percentages)],
        title="Campaigns",
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        fontsize=8,
        title_fontsize=9
    )
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    return fig

def draw_roas_split_bar_chart(roas_series):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.barh(roas_series.index, roas_series.values, color="#007fff", height=0.4)

    ax.set_xlabel("ROAS")
    ax.set_title("ROAS Split by Adset")

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                f"{width:.2f}", va='center', fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig

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

    #1. Cost Split (Donut) - only if we have data  ---- new change
    # fig1 = draw_donut_chart(top_spend.values, top_spend.index, "Cost Split")
    # figs.append(("Cost Split", generate_chart_image(fig1)))
    # fig2 = draw_donut_chart(top_revenue.values, top_revenue.index, "Revenue Split")
    # figs.append(("Revenue Split", generate_chart_image(fig2)))
    if not top_spend.empty and top_spend.values.sum() > 0 and np.all(np.isfinite(top_spend.values)):
        fig1 = draw_donut_chart(top_spend.values, top_spend.index, "Cost Split")
        figs.append(("Cost Split", generate_chart_image(fig1)))
    else:
        print("⚠️ Skipping Cost Split chart — no valid spend data.")

    if not top_revenue.empty and top_revenue.values.sum() > 0 and np.all(np.isfinite(top_revenue.values)):
        fig2 = draw_donut_chart(top_revenue.values, top_revenue.index, "Revenue Split")
        figs.append(("Revenue Split", generate_chart_image(fig2)))
    else:
        print("⚠️ Skipping Revenue Split chart — no valid revenue data.")


    
   
   # 3. ROAS Split (Horizontal bar)    
    if not top_roas.empty:
        fig3, ax3 = plt.subplots(figsize=(7, 4))  # Wider figure (was 5.5)
        # Get max ROAS value and add 25% padding
        max_val = top_roas.max() 
        x_limit = max_val * 1.25
        ax3.barh(
            top_roas.index[::-1],
            top_roas.values[::-1],
            color='#ff00aa',
            height=0.5  # Keep bar thickness same
        )
        # Critical change - set axis limits to maximize bar lengths
        ax3.set_xlim(0, x_limit)
    
        # Adjust layout to prevent cutting off
        plt.subplots_adjust(left=0.3, right=0.95)  # More space for labels
    
        ax3.set_title('ROAS Split', fontsize=12)
        ax3.set_xlabel("ROAS", fontsize=10)  # Added fontsize
        ax3.tick_params(axis='both', labelsize=10)
        ax3.yaxis.label.set_size(10)
        plt.tight_layout()
        figs.append(("ROAS Split", generate_chart_image(fig3)))
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

def generate_revenue_by_campaign_chart(df):
    

    df['date'] = pd.to_datetime(df['date'])
    df['purchase_value'] = pd.to_numeric(df['purchase_value'], errors='coerce').fillna(0)
    df['campaign_name'] = df['campaign_name'].fillna("Unknown Campaign")

    grouped = df.groupby(['campaign_name', 'date'])['purchase_value'].sum().reset_index()
    pivot_df = grouped.pivot(index='date', columns='campaign_name', values='purchase_value').fillna(0)

    fig, ax = plt.subplots(figsize=(15, 6), dpi=200)

    for column in pivot_df.columns:
        ax.plot(pivot_df.index, pivot_df[column], label=column, linewidth=1.5, marker='o', markersize=3)

    ax.set_title("Revenue By Campaigns", fontsize=16, weight='bold')
    ax.set_ylabel("Revenue")
    ax.set_xlabel("Day")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))

    fig.tight_layout()
    return ("Revenue By Campaigns", generate_chart_image(fig))

async def generate_roas_summary_text(full_df: pd.DataFrame, currency_symbol: str) -> str:
    from services.prompts import RESULTS_SETUP_PROMPT
    from services.deepseek_audit import generate_llm_content

    # Reduce data to relevant fields
    if full_df.empty or 'campaign_name' not in full_df.columns:
        return "⚠️ No campaign data available to summarize."

    df = full_df[['campaign_name', 'spend', 'purchase_value', 'purchases']].copy()
    df = df[df['campaign_name'].notna()]
    df['roas'] = df['purchase_value'] / df['spend'].replace(0, 1)
    df['cpa'] = df['spend'] / df['purchases'].replace(0, 1)

    # Sort by performance
    df = df.sort_values('roas', ascending=False)
    df['spend'] = df['spend'].round(2)
    df['purchase_value'] = df['purchase_value'].round(2)
    df['roas'] = df['roas'].round(2)
    df['cpa'] = df['cpa'].round(2)

    # Convert to dict for prompt
    records = df.to_dict(orient='records')
    summary_data = {
        "summary_metrics": {
            "total_spend": float(df['spend'].sum()),
            "total_revenue": float(df['purchase_value'].sum()),
            "total_purchases": int(df['purchases'].sum()),
            "avg_roas": float(df['roas'].mean()),
            "avg_cpa": float(df['cpa'].mean())
        },
        "campaigns": records
    }

    prompt = (
        f"Write a concise 1-paragraph summary about Meta Ads campaign performance. "
        f"Use {currency_symbol} for monetary values. "
        f"Include total spend, revenue, purchases, average ROAS, and CPA. "
        f"Mention 1-2 top-performing campaigns (with high ROAS) and 1-2 poor ones (low ROAS or high CPA). "
        f"Conclude with 1 recommendation to improve overall performance."
        
    )

    return await generate_llm_content(prompt, summary_data)

async def generate_adset_summary(full_df: pd.DataFrame, currency_symbol: str) -> str:
    from services.prompts import RESULTS_SETUP_PROMPT
    from services.deepseek_audit import generate_llm_content

    if full_df.empty or 'adset_name' not in full_df.columns:
        return "⚠️ No ad set data available to summarize."

    df = full_df[['adset_name', 'spend', 'purchase_value', 'purchases']].copy()
    df = df[df['adset_name'].notna()]
    df['spend'] = pd.to_numeric(df['spend'], errors='coerce').fillna(0)
    df['purchase_value'] = pd.to_numeric(df['purchase_value'], errors='coerce').fillna(0)
    df['purchases'] = pd.to_numeric(df['purchases'], errors='coerce').fillna(0)
    df['roas'] = df['purchase_value'] / df['spend'].replace(0, 1)
    df['cpa'] = df['spend'] / df['purchases'].replace(0, 1)

    df = df.sort_values('roas', ascending=False)
    df['spend'] = df['spend'].round(2)
    df['purchase_value'] = df['purchase_value'].round(2)
    df['roas'] = df['roas'].round(2)
    df['cpa'] = df['cpa'].round(2)

    records = df.to_dict(orient='records')

    summary_data = {
        "summary_metrics": {
            "total_spend": float(df['spend'].sum()),
            "total_revenue": float(df['purchase_value'].sum()),
            "total_purchases": int(df['purchases'].sum()),
            "avg_roas": float(df['roas'].mean()),
            "avg_cpa": float(df['cpa'].mean())
        },
        "adsets": records
    }

    prompt = (
        f"Write a detailed summary of Meta Ads *ad set performance* in about 150–200 words. "
        f"Use {currency_symbol} for all money values.\n\n"
        f"1. Identify and mention top-performing ad sets (with ROAS > 2.5 and CPA < {currency_symbol}500).\n"
        f"2. Point out poor-performing ad sets (ROAS < 0.5 or CPA > {currency_symbol}5000).\n"
        f"3. Summarize trends (like remarketing, lookalikes, interest-based, funnel stage, etc.).\n"
        f"4. Give 1–2 practical recommendations based on insights.\n\n"
        f"Make it professional and concise. Mention adset names. Avoid listing all rows."
    )

    return await generate_llm_content(prompt, summary_data)

async def generate_ad_summary(full_df: pd.DataFrame, currency_symbol: str) -> str:
    from services.deepseek_audit import generate_llm_content

    if full_df.empty or 'ad_name' not in full_df.columns:
        return "⚠️ No ad data available to summarize."

    df = full_df[['ad_name', 'spend', 'purchase_value', 'purchases']].copy()
    df = df[df['ad_name'].notna()]
    df['spend'] = pd.to_numeric(df['spend'], errors='coerce').fillna(0)
    df['purchase_value'] = pd.to_numeric(df['purchase_value'], errors='coerce').fillna(0)
    df['purchases'] = pd.to_numeric(df['purchases'], errors='coerce').fillna(0)
    df['roas'] = df['purchase_value'] / df['spend'].replace(0, 1)
    df['cpa'] = df['spend'] / df['purchases'].replace(0, 1)

    df = df.sort_values('roas', ascending=False)
    df['spend'] = df['spend'].round(2)
    df['purchase_value'] = df['purchase_value'].round(2)
    df['roas'] = df['roas'].round(2)
    df['cpa'] = df['cpa'].round(2)

    records = df.to_dict(orient='records')

    summary_data = {
        "summary_metrics": {
            "total_spend": float(df['spend'].sum()),
            "total_revenue": float(df['purchase_value'].sum()),
            "total_purchases": int(df['purchases'].sum()),
            "avg_roas": float(df['roas'].mean()),
            "avg_cpa": float(df['cpa'].mean())
        },
        "ads": records
    }

    prompt = (
        f"Write a detailed summary of Meta Ads *ad-level performance* in about 150–200 words. "
        f"Use {currency_symbol} for all money values.\n\n"
        f"1. Highlight top-performing ads (ROAS > 3.0 and CPA < {currency_symbol}400).\n"
        f"2. Identify underperforming ads (ROAS < 0.4 or CPA > {currency_symbol}5000).\n"
        f"3. Mention creative insights or copy performance patterns if possible (like UGC, hooks, etc).\n"
        f"4. Provide 1–2 sharp, actionable recommendations for improvement.\n\n"
        f"Make it insightful and executive-friendly. Mention ad names selectively — don’t list everything."
    )

    return await generate_llm_content(prompt, summary_data)

async def fetch_demographic_insights(account_id: str, access_token: str):
    url = f"https://graph.facebook.com/v22.0/{account_id}/insights"
    now = datetime.now()
    since = (now - timedelta(days=30)).strftime('%Y-%m-%d')
    until = now.strftime('%Y-%m-%d')

    params = {
        "fields": "spend,impressions,clicks,reach,actions,action_values",
        "level": "ad",
        "breakdowns": "age,gender",
        "time_range": json.dumps({"since": since, "until": until}),
        "access_token": access_token
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json().get("data", [])
        print("📦 Raw demographic data:", json.dumps(data, indent=2)) 
        df = pd.DataFrame(data)

        if df.empty:
            return df

        # Preprocess data
        df['spend'] = pd.to_numeric(df['spend'], errors='coerce').fillna(0)
        #df['purchases'] = df['actions'].apply(lambda acts: next((float(a.get('value')) for a in acts if a.get("action_type") == "purchase"), 0))
        def extract_purchase(actions):
            if isinstance(actions, list):
                for a in actions:
                    if isinstance(a, dict) and a.get("action_type") == "purchase":
                        return float(a.get("value", 0))
            return 0

        df['purchases'] = df['actions'].apply(extract_purchase)
        
        def extract_purchase_value(action_values):
            if isinstance(action_values, list):
                for a in action_values:
                    if isinstance(a, dict) and a.get("action_type") == "purchase":
                        return float(a.get("value", 0))
            return 0

        df['purchase_value'] = df['action_values'].apply(extract_purchase_value)


        #df['purchase_value'] = df['action_values'].apply(lambda acts: next((float(a.get('value')) for a in acts if a.get("action_type") == "purchase"), 0))
        df['cpa'] = df['spend'] / df['purchases'].replace(0, 1)
        df['roas'] = df['purchase_value'] / df['spend'].replace(0, 1)

        return df[['age', 'gender', 'spend', 'purchases', 'purchase_value', 'cpa', 'roas']]

async def fetch_facebook_insights(page_id: str, page_token: str):
    """Fetch Facebook page insights"""
    try:
        base_url = f"https://graph.facebook.com/v22.0/{page_id}/insights"
        since = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
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
    timeout = httpx.Timeout(60.0, connect=10.0)

    try:
        url = f"https://graph.facebook.com/v22.0/me/adaccounts"
        async with httpx.AsyncClient(timeout=timeout) as client:
            acc_resp = await client.get(url, params={
                "access_token": user_token,
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
                if 'adsets' not in acc or not acc['adsets'].get('data'):
                    print(f"⚠️ No adsets found for account {acc.get('name')}")
                    continue
                print(f"✅ Processing account: {acc.get('name')} (ID: {acc.get('id')})")


                print(f"🔍 Processing account: {acc.get('name')} ({acc.get('id')})")

                ad_url = f"https://graph.facebook.com/v22.0/{acc['id']}/insights"
                now = datetime.now(timezone.utc)
                safe_until = (now - timedelta(days=2)).strftime("%Y-%m-%d")
                safe_since = (now - timedelta(days=32)).strftime("%Y-%m-%d")
                print(f"📅 Fetching data from {safe_since} to {safe_until}")
                
                


                params = {
                    "fields": "campaign_name,adset_name,ad_name,spend,impressions,clicks,cpc,ctr,actions,action_values,date_start,account_currency,account_name",
                    "time_range": json.dumps({"since": safe_since, "until": safe_until}),
                    "time_increment": 1,
                    "level": "ad",
                    "access_token": user_token
                }
                
                reach_params = {
                    "fields": "adset_id,reach,date_start",  # minimum fields needed
                    "time_range": json.dumps({"since": safe_since, "until": safe_until}),
                    "time_increment": 1,
                    "level": "adset",  # ✅ Fetch reach from adset level
                    "access_token": user_token
                }
                
                demographic_params = {
                    "fields": "age,gender,adset_id,spend,impressions,reach",
                    "breakdowns": "age,gender",
                    "time_range": json.dumps({"since": safe_since, "until": safe_until}),
                    "time_increment": 1,
                    "level": "ad",
                    "access_token": user_token
                }



                ad_results = []
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(ad_url, params=params)
                    if "error" in response.json():
                        print("⚠️ Graph API Error:", response.json()["error"]["message"])

                    response.raise_for_status()
                    data_page = response.json()
                    ad_results.extend(data_page.get("data", []))

                    # 👇 PAGINATE over `paging.next`
                    while data_page.get("paging", {}).get("next"):
                        next_url = data_page["paging"]["next"]
                        next_response = await client.get(next_url, follow_redirects=True)
                        next_response.raise_for_status()
                        data_page = next_response.json()
                        ad_results.extend(data_page.get("data", []))
                    print(f"✅ Fetched {len(ad_results)} ad results for account {acc.get('name')} ({acc.get('id')})")

                        
                    # 🔍 Fetch reach at adset level (for fatigue analysis)
                    reach_url = f"https://graph.facebook.com/v22.0/{acc['id']}/insights"

                    reach_params = {
                        "fields": "adset_id,reach,date_start",
                        "time_range": json.dumps({"since": safe_since, "until": safe_until}),
                        "time_increment": 1,
                        "level": "adset",
                        "access_token": user_token
                    }
                    
                    # demographic_params = {
                    #     "fields": "adset_id,age,gender,spend,impressions,reach,date_start",
                    #     "breakdowns": "age,gender",
                    #     "time_range": json.dumps({"since": safe_since, "until": safe_until}),
                    #     "time_increment": 1,
                    #     "level": "ad",
                    #     "access_token": user_token
                    # }

                    reach_df = pd.DataFrame()
                    try:
                        reach_response = await client.get(reach_url, params=reach_params)
                        reach_response.raise_for_status()
                        reach_data = reach_response.json().get("data", [])
                        reach_df = pd.DataFrame(reach_data)
                    except Exception as e:
                        print(f"⚠️ Failed to fetch reach data for account {acc['id']}: {e}")
                        
                    # 🔍 Fetch demographic data
                    # demographic_url = f"https://graph.facebook.com/v22.0/{acc['id']}/insights"
                    # demographic_df = pd.DataFrame()
                    # try:
                    #     demo_response = await client.get(demographic_url, params=demographic_params)
                    #     demo_response.raise_for_status()
                    #     demo_data = demo_response.json().get("data", [])
                    #     demographic_df = pd.DataFrame(demo_data)
                    #     print(f"✅ Fetched demographic data for {acc['id']}, shape: {demographic_df.shape}")
                    # except Exception as e:
                    #     print(f"⚠️ Failed to fetch demographic data for account {acc['id']}: {e}")
                    #     demographic_df = pd.DataFrame()
 
                        
                    # ✅ DEBUG: Print full data sample after all pages
                    print("📦 Final sample of fetched ad data (first 3 rows):")
                    import pprint
                    pprint.pprint(ad_results[:3], indent=2)

                print(f"✅ Total insights for account {acc['id']}: {len(ad_results)}")
                # 🧠 Merge reach into ad-level data if available
                if not reach_df.empty:
                    reach_df["date_start"] = pd.to_datetime(reach_df["date_start"])
                    for ad in ad_results:
                        if "adset_id" in ad and "date_start" in ad:
                            match = reach_df[
                                (reach_df["adset_id"] == ad["adset_id"]) &
                                (pd.to_datetime(reach_df["date_start"]) == pd.to_datetime(ad["date_start"]))
                            ]
                            if not match.empty:
                                ad["reach"] = match["reach"].values[0]
                demographic_df = pd.DataFrame()
                                
                if not demographic_df.empty:
                    demographic_df["reach"] = pd.to_numeric(demographic_df["reach"], errors='coerce').fillna(0)
                    demographic_df["spend"] = pd.to_numeric(demographic_df["spend"], errors='coerce').fillna(0)
                    demographic_df["impressions"] = pd.to_numeric(demographic_df["impressions"], errors='coerce').fillna(0)

                    # Optional: clean or rename columns if needed
                    demographic_df = demographic_df.rename(columns={"date_start": "date"})

                    # If needed: merge into original_df or use separately
                    print("📊 Demographic breakdown sample:")
                    print(demographic_df.head())



                for ad in ad_results:
                    if 'account_currency' not in ad:
                        ad["account_currency"] = acc.get("account_currency", "USD")
                    ad["account_id"] = acc.get("id") # <-- ADD THIS LINE
                    insights_data.append(ad)
                    
                demographic_df = pd.DataFrame()

                if not demographic_df.empty:
                    demographic_df["reach"] = pd.to_numeric(demographic_df["reach"], errors='coerce').fillna(0)
                    demographic_df["spend"] = pd.to_numeric(demographic_df["spend"], errors='coerce').fillna(0)
                    demographic_df["impressions"] = pd.to_numeric(demographic_df["impressions"], errors='coerce').fillna(0)

                    demographic_df = demographic_df.rename(columns={"date_start": "date"})

                    print("📊 Demographic breakdown sample:")
                    print(demographic_df.head())

            print(f"📦 Fetched total {len(insights_data)} ads across all accounts.")
            
            return insights_data

    except Exception as e:
        print(f"❌ Error in fetch_ad_insights: {str(e)}")
        return []



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
    from services.generate_pdf import generate_pdf_report
    from services.deepseek_audit import fetch_demographic_insights
    """Generate audit report and return PDF"""
    try:
        print("🔄 Starting audit generation...")

        if not DEEPSEEK_API_URL or not DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_URL and DEEPSEEK_API_KEY environment variables must be set")

        print("📊 Fetching Facebook data...")
        page_data = await fetch_facebook_insights(page_id, page_token)
        ad_data  = await fetch_ad_insights(user_token)
        print("🔍 ad_data structure:", type(ad_data))
        #account_id = ad_data[0]['account_id'] if ad_data else None
        account_id = None
        if ad_data and isinstance(ad_data, list) and isinstance(ad_data[0], dict):
            account_id = ad_data[0].get('account_id')
        elif isinstance(ad_data, list):
            for item in ad_data:
                if isinstance(item, dict) and 'account_id' in item:
                    account_id = item['account_id']
                    break
        print(f"🆔 Extracted Account ID: {account_id}")
        
        demographic_df = pd.DataFrame() # Default to empty DataFrame
        if account_id:
            demographic_df = await fetch_demographic_insights(account_id, user_token)
        else:
            print("⚠️ Could not determine account_id. Skipping demographic insights fetch.")
        
        #demographic_df = await fetch_demographic_insights(account_id, user_token)
        
        if not demographic_df.empty:
            demographic_df['spend'] = pd.to_numeric(demographic_df['spend'], errors='coerce').fillna(0)
            demographic_df['impressions'] = pd.to_numeric(demographic_df['impressions'], errors='coerce').fillna(0)
            demographic_df['reach'] = pd.to_numeric(demographic_df['reach'], errors='coerce').fillna(0)
            
            #Rename columns to match expected format
            

            demographic_grouped = demographic_df.groupby(['Age', 'Gender']).agg({
                "spend": "sum",
                "reach": "sum",
                "impressions": "sum"
            }).reset_index()
            print("✅ Grouped demographic data:")
            print(demographic_grouped.head())


        # Filter out invalid entries
        ad_data = [d for d in ad_data if isinstance(d, dict) and 'date_start' in d and d.get('date_start')]
        ad_raw = []        
        
        if not ad_data:
            print("🚨 Filtered ad_data is empty. Attempting fallback fetch...")
            ad_raw = await fetch_ad_insights(user_token)
            print("🔍 ad_data structure:", type(ad_raw))
            print("🔍 Raw ad data preview:", ad_raw[:2])
            
            #ad_data = [d for d in ad_raw if isinstance(d, dict) and 'date_start' in d and d.get('date_start')]

        if not ad_data:
            print("🚨 No ad data returned initially. Attempting fallback fetch...")
            ad_data = await fetch_ad_insights(user_token)
        ad_data = [d for d in ad_data if isinstance(d, dict) and 'date_start' in d and d.get('date_start')]
        
        # Final check
        if not ad_data:
            print("❌ No usable ad entries with 'date_start'.")
            raise ValueError("❌ All ad insights entries are missing 'date_start' — cannot proceed.")
        
        PURCHASE_KEYS = [
            "offsite_conversion.purchase",
            "offsite_conversion.fb_pixel_purchase",
            "offsite_conversion.fb_pixel_custom",
            "offsite_conversion.custom.1408006162945363",
            "offsite_conversion.custom.587738624322885",
            "purchase"
        ]

        for ad in ad_data:
            try:
                actions = {d["action_type"]: float(d["value"]) for d in ad.get("actions", []) if "action_type" in d and "value" in d}
                values = {d["action_type"]: float(d["value"]) for d in ad.get("action_values", []) if "action_type" in d and "value" in d}
            except Exception as e:
                print(f"⚠️ Error parsing actions in ad: {e}")
                actions, values = {}, {}
            ad["purchases"] = sum(actions.get(k, 0) for k in PURCHASE_KEYS)

            #ad["purchases"] = sum(actions.get(k, 0) for k in PURCHASE_KEYS)
            raw_value = sum(values.get(k, 0) for k in PURCHASE_KEYS)
            if raw_value == 0 and ad["purchases"] > 0:
                # Assume 1000 per purchase as fallback (adjust as needed)
                raw_value = ad["purchases"] * 1000
                ad["purchase_value"] = raw_value
            
            # ad["purchase_value"] = sum(
            #     float(d.get("value", 0))
            #     for d in ad.get("action_values", [])
            #     if d.get("action_type") in PURCHASE_KEYS
            # )

            ad["link_clicks"] = actions.get("link_click", 0)
                # ✅ Ensure non-missing values for charts and grouping
            if "purchase_value" not in ad or not isinstance(ad["purchase_value"], (int, float)):
                ad["purchase_value"] = 0
            if "purchases" not in ad or not isinstance(ad["purchases"], (int, float)):
                ad["purchases"] = 0




        # Create original DataFrame with date_start intact
        original_df = pd.DataFrame(ad_data)
        original_df['campaign_name'] = original_df['campaign_name'].fillna("Unknown Campaign")
        original_df['adset_name'] = original_df['adset_name'].fillna("Unknown Adset")

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

        cutoff = pd.Timestamp.today() - pd.Timedelta(days=30)
        ad_insights_df = grouped_df[grouped_df['date'] >= cutoff].copy()

        if ad_insights_df.empty:
            print("⚠️ No data in last 30 days. Using last available 30 records.")
            ad_insights_df = grouped_df.tail(30)

        currency = "USD"  # Default
        currency_symbol = "$"

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
        
        # ✅ Log final data sample for verification
        print("📊 Sample of original_df:")
        print(original_df[["date", "campaign_name", "spend", "purchase_value", "purchases"]].tail(5))

        print("📄 Generating PDF report...")
        pdf_response = generate_pdf_report(
            sections,
            ad_insights_df=ad_insights_df,
            full_ad_insights_df=original_df,
            currency_symbol=currency_symbol,
            split_charts=split_charts,
            demographic_df=demographic_df
            
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
#------------------------------------------------------------------------------------