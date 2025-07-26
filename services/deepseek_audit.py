# services/deepseek_audit.py
import httpx
import os
import requests
from datetime import datetime, timedelta, timezone
import json
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from io import BytesIO
import base64
import matplotlib.ticker as mticker
import matplotlib.patheffects as path_effects
import pprint # For pretty printing
from services.prompts import EXECUTIVE_SUMMARY_PROMPT, ACCOUNT_NAMING_STRUCTURE_PROMPT, TESTING_ACTIVITY_PROMPT, REMARKETING_ACTIVITY_PROMPT, RESULTS_SETUP_PROMPT


logger = logging.getLogger(__name__)

DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# --- (Chart Generation Functions - No changes needed here, keeping for context) ---

def generate_chart_1(ad_insights_df):
    fig, ax1 = plt.subplots(figsize=(20, 8), dpi=200, constrained_layout=True)
    ad_insights_df['purchase_value'] = ad_insights_df['purchase_value'].fillna(0)
    ad_insights_df['spend'] = ad_insights_df['spend'].fillna(0)
    ax2 = ax1.twinx()
    ax2.bar(
        ad_insights_df["date"], ad_insights_df["purchase_value"],
        color="#B2FF59", edgecolor="#76FF03", width=0.8, label="Purchase Conversion Value", alpha=0.9, zorder=2
    )
    ax2.plot(
        ad_insights_df["date"], ad_insights_df["spend"],
        color="magenta", marker="o", label="Amount Spent", linewidth=2.5, zorder=3
    )
    ax1.set_ylabel("Purchases", color="#4CAF50", fontsize=16)
    ax2.set_ylabel("Amount Spent", color="magenta", fontsize=16)
    ax1.tick_params(axis='y', labelcolor="#4CAF50", labelsize=14)
    ax2.tick_params(axis='y', labelcolor="magenta", labelsize=10)
    ax1.set_xticks(ad_insights_df["date"])
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax1.tick_params(axis='x', rotation=45, labelsize=14)
    ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    return fig

def generate_chart_image(fig):
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=200)
    buf.seek(0)
    plt.close(fig)
    return buf



def generate_key_metrics_section(ad_insights_df, currency_symbol="₹"):
    if ad_insights_df.empty or len(ad_insights_df) < 2:
        print("⚠️ Not enough data to generate charts.")
        return "No data available for Key Metrics.", []

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
    chart_imgs = []

    # Chart 0: Purchases vs Amount Spent
    fig0 = generate_chart_1(ad_insights_df)
    chart_imgs.append(("Purchases vs Amount Spent", generate_chart_image(fig0)))

    # Chart 1: Purchases vs ROAS
    purchases_df = ad_insights_df.sort_values("date")[-30:]
    fig1, ax1 = plt.subplots(figsize=(18, 8))
    ax1.bar(purchases_df["date"], purchases_df["purchases"], width=0.9, color="#0d0c42", label="Purchases")
    ax1.set_ylabel("Purchases", color="#0d0c42", fontsize=16)
    ax1.tick_params(axis='y', labelcolor="#0d0c42", labelsize=14)
    ax2 = ax1.twinx()
    ax2.plot(purchases_df["date"], purchases_df["roas"], color="#ff00aa", marker="o", label="ROAS")
    ax2.set_ylabel("ROAS", color="#ff00aa", fontsize=16)
    ax2.tick_params(axis='y', labelcolor="#ff00aa", labelsize=14)

    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.xticks(rotation=45, fontsize=14)

    # for spine in ax1.spines.values(): spine.set_visible(False)
    # for spine in ax2.spines.values(): spine.set_visible(False)
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("grey")
        spine.set_linewidth(1)

    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("grey")
        spine.set_linewidth(1)


    ax1.legend(loc='upper left', bbox_to_anchor=(0.01, 1.15), frameon=False,fontsize=12)
    ax2.legend(loc='upper right', bbox_to_anchor=(0.99, 1.15), frameon=False,fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.3)

    fig1.patch.set_facecolor('white')
    fig1.patch.set_alpha(1)
    fig1.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.2)
    fig1.patch.set_linewidth(0)
    fig1.patch.set_edgecolor('none')

    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    plt.tight_layout(h_pad=3.0)
    chart_imgs.append(("Purchases vs ROAS", generate_chart_image(fig1)))

    # Chart 2: CPA vs Link CPC (Line Chart)
    cpa_df = ad_insights_df.sort_values("date")[-30:]
    fig2, ax3 = plt.subplots(figsize=(18, 8))
    ax3.plot(cpa_df["date"], cpa_df["cpa"], color="#2079b5", linewidth=2, marker='o', label="CPA")
    ax3.set_ylabel("CPA", color="#2079b5", fontsize=16)
    ax3.tick_params(axis='y', labelcolor="#2079b5", labelsize=14)
    ax4 = ax3.twinx()
    ax4.plot(cpa_df["date"], cpa_df["cpc"], color="#b3e08b", linewidth=2, marker='o', label="Link CPC")
    ax4.set_ylabel("Link CPC", color="#b3e08b", fontsize=16)
    ax4.tick_params(axis='y', labelcolor="#b3e08b", labelsize=14)

    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.xticks(rotation=45, fontsize=14)

    for spine in ax3.spines.values(): spine.set_visible(False)
    for spine in ax4.spines.values(): spine.set_visible(False)

    ax3.legend(loc='upper left', bbox_to_anchor=(0.01, 1.15), frameon=False,fontsize=12)
    ax4.legend(loc='upper right', bbox_to_anchor=(0.99, 1.15), frameon=False,fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.3)

    fig2.patch.set_facecolor('white')
    fig2.patch.set_alpha(1)
    fig2.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.2)
    fig2.patch.set_linewidth(0)
    fig2.patch.set_edgecolor('none')
    plt.tight_layout(h_pad=3.0)
    chart_imgs.append(("CPA vs Link CPC", generate_chart_image(fig2)))

    # Chart 3: Click to Conversion vs CTR
    click_df = ad_insights_df.sort_values("date")[-30:]
    fig3, ax5 = plt.subplots(figsize=(18, 8))
    bars = ax5.bar(click_df["date"], click_df["click_to_conversion"], width=0.8, color="#0000ff", label="Click to Conversion")
    ax5.set_ylabel("Click to Conversion", color="#0000ff", fontsize=16)
    ax5.tick_params(axis='y', labelcolor="#0000ff", labelsize=14)
    ax6 = ax5.twinx()
    ax6.plot(click_df["date"], click_df["ctr"], color="#f8a83c", marker="o", label="CTR")
    ax6.set_ylabel("CTR", color="#f8a83c", fontsize=16)
    ax6.tick_params(axis='y', labelcolor="#f8a83c", labelsize=14)

    ax5.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.xticks(rotation=45, fontsize=14)

    for spine in ax5.spines.values(): spine.set_visible(False)
    for spine in ax6.spines.values(): spine.set_visible(False)

    ax5.legend(loc='upper left', bbox_to_anchor=(0.01, 1.15), frameon=False,fontsize=12)
    ax6.legend(loc='upper right', bbox_to_anchor=(0.99, 1.15), frameon=False,fontsize=12)
    ax5.grid(True, linestyle='--', alpha=0.3)

    for bar in bars:
        bar.set_linewidth(0)
        bar.set_path_effects([
            path_effects.SimplePatchShadow(offset=(2, -2), alpha=0.2),
            path_effects.Normal()
        ])

    fig3.patch.set_facecolor('white')
    fig3.patch.set_alpha(1)
    fig3.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.25)
    fig3.patch.set_linewidth(0)
    fig3.patch.set_edgecolor('none')
    plt.tight_layout(h_pad=3.0)
    chart_imgs.append(("Click to Conversion vs CTR", generate_chart_image(fig3)))

    table_html = ad_insights_df.to_html(index=False, border=0)
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
    if 'reach' not in df.columns:
        print("⚠️ 'reach' column missing, using 'impressions' as fallback.")
        df['reach'] = df['impressions']
    df['reach'] = pd.to_numeric(df['reach'], errors='coerce').fillna(1)
    df['spend'] = pd.to_numeric(df['spend'], errors='coerce').fillna(0)
    df['purchase_value'] = pd.to_numeric(df['purchase_value'], errors='coerce').fillna(0)
    df['purchases'] = pd.to_numeric(df['purchases'], errors='coerce').fillna(0)
    df['frequency'] = df['impressions'] / df['reach'].replace(0, 1)
    df['roas'] = df['purchase_value'] / df['spend'].replace(0, 1)
    #df['cpa'] = df['spend'] / df['purchases'].replace(0, 1) NA
    # Ensure purchases are numeric and handle NaN
    df['purchases'] = pd.to_numeric(df['purchases'], errors='coerce').fillna(0)
    df['cpa'] = df.apply(lambda row: row['spend'] / row['purchases'] if row['purchases'] > 0 else np.nan, axis=1)
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
    def truncate_label(label, max_words=4):
        tokens = label.split()
        return " ".join(tokens[:max_words]) + "..." if len(tokens) > max_words else label
    truncated_labels = [truncate_label(label) for label in labels]
    percentages = 100 * values / values.sum()
    color_map = plt.get_cmap('tab20c')
    colors = color_map.colors[:len(truncated_labels)]
    fig, ax = plt.subplots(figsize=(5, 5))  
    wedges, _ = ax.pie(
        values,
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.4)
    )
    ax.text(0, 0, "100%", ha='center', va='center', fontsize=14, weight='bold')
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
        currency_symbol = "₹"

    grouped = df[df['campaign_name'].notna()].copy()
    grouped['spend'] = pd.to_numeric(grouped['spend'], errors='coerce').fillna(0)
    grouped['purchase_value'] = pd.to_numeric(grouped['purchase_value'], errors='coerce').fillna(0)

    if grouped.empty or grouped['spend'].sum() == 0:
        print("⚠️ No valid campaign data available for split charts")
        return []

    spend_split = grouped.groupby('campaign_name')['spend'].sum().sort_values(ascending=False)
    revenue_split = grouped.groupby('campaign_name')['purchase_value'].sum().sort_values(ascending=False)
    roas_split = revenue_split / spend_split.replace(0, 1)
    roas_split = roas_split.dropna()
    top_spend = spend_split.head(8) if not spend_split.empty else pd.Series(dtype=float)
    top_revenue = revenue_split.head(8) if not revenue_split.empty else pd.Series(dtype=float)
    top_roas = roas_split.sort_values(ascending=False).head(10) if not roas_split.empty else pd.Series(dtype=float)
    figs = []

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
   
    if not top_roas.empty:
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        max_val = top_roas.max() 
        x_limit = max_val * 1.25
        ax3.barh(
            top_roas.index[::-1],
            top_roas.values[::-1],
            color='#ff00aa',
            height=0.5
        )
        plt.subplots_adjust(left=0.3, right=0.95)
        ax3.set_xlim(0, x_limit)
        ax3.set_title('ROAS Split', fontsize=12)
        ax3.set_xlabel("ROAS", fontsize=10)
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
    from services.prompts import RESULTS_SETUP_PROMPT # No change, keep as is
    from services.deepseek_audit import generate_llm_content # No change, keep as is

    if full_df.empty or 'campaign_name' not in full_df.columns:
        return "⚠️ No campaign data available to summarize."

    df = full_df[['campaign_name', 'spend', 'purchase_value', 'purchases']].copy()
    df = df[df['campaign_name'].notna()]
    df['roas'] = df['purchase_value'] / df['spend'].replace(0, 1)
    #df['cpa'] = df['spend'] / df['purchases'].replace(0, 1) NA
    # Ensure purchases are numeric and handle NaN
    df['purchases'] = pd.to_numeric(df['purchases'], errors='coerce').fillna(0)
    df['cpa'] = df.apply(lambda row: row['spend'] / row['purchases'] if row['purchases'] > 0 else np.nan, axis=1)
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
    from services.prompts import RESULTS_SETUP_PROMPT # No change, keep as is
    from services.deepseek_audit import generate_llm_content # No change, keep as is

    if full_df.empty or 'adset_name' not in full_df.columns:
        return "⚠️ No ad set data available to summarize."

    df = full_df[['adset_name', 'spend', 'purchase_value', 'purchases']].copy()
    df = df[df['adset_name'].notna()]
    df['spend'] = pd.to_numeric(df['spend'], errors='coerce').fillna(0)
    df['purchase_value'] = pd.to_numeric(df['purchase_value'], errors='coerce').fillna(0)
    df['purchases'] = pd.to_numeric(df['purchases'], errors='coerce').fillna(0)
    df['roas'] = df['purchase_value'] / df['spend'].replace(0, 1)
    #df['cpa'] = df['spend'] / df['purchases'].replace(0, 1) NA
    # Ensure purchases are numeric and handle NaN
    df['purchases'] = pd.to_numeric(df['purchases'], errors='coerce').fillna(0)
    df['cpa'] = df.apply(lambda row: row['spend'] / row['purchases'] if row['purchases'] > 0 else np.nan, axis=1)
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
    from services.deepseek_audit import generate_llm_content # No change, keep as is

    if full_df.empty or 'ad_name' not in full_df.columns:
        return "⚠️ No ad data available to summarize."

    df = full_df[['ad_name', 'spend', 'purchase_value', 'purchases']].copy()
    df = df[df['ad_name'].notna()]
    df['spend'] = pd.to_numeric(df['spend'], errors='coerce').fillna(0)
    df['purchase_value'] = pd.to_numeric(df['purchase_value'], errors='coerce').fillna(0)
    df['purchases'] = pd.to_numeric(df['purchases'], errors='coerce').fillna(0)
    df['roas'] = df['purchase_value'] / df['spend'].replace(0, 1)
    #df['cpa'] = df['spend'] / df['purchases'].replace(0, 1)
    # Ensure purchases are numeric and handle NaN
    df['purchases'] = pd.to_numeric(df['purchases'], errors='coerce').fillna(0)
    df['cpa'] = df.apply(lambda row: row['spend'] / row['purchases'] if row['purchases'] > 0 else np.nan, axis=1)
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
        logger.info(f"📦 Raw demographic data (first 2 entries): {json.dumps(data[:2], indent=2)}")
        print("📦 Raw demographic data:", json.dumps(data, indent=2))

        # Create DataFrame from fetched data. If data is empty, it will be an empty DataFrame.
        df = pd.DataFrame(data)

        # 🛡 Ensure required columns exist, filling with default values if missing
        required_breakdown_cols = ['age', 'gender']
        for col in required_breakdown_cols:
            if col not in df.columns:
                df[col] = 'unknown' # Default to 'unknown' if breakdown column is entirely missing
            else:
                df[col] = df[col].astype(str).fillna('unknown') # Ensure string type and fill NaN

        # 🧩 Fill missing complex columns with empty lists for consistent processing
        for col in ['actions', 'action_values']:
            if col not in df.columns:
                df[col] = [[] for _ in range(len(df))] # Ensure column exists with empty lists

        # 🧼 Filter out "unknown" entries in age and gender (case-insensitive)
        df = df[(df['age'].str.lower().str.strip() != 'unknown') &
                (df['gender'].str.lower().str.strip() != 'unknown')]

        # If after filtering, the DataFrame is empty, return an empty structured DataFrame
        if df.empty:
            logger.warning("Demographic data is empty or all entries are 'unknown' after filtering.")
            return pd.DataFrame(columns=['age', 'gender', 'spend', 'purchases', 'purchase_value', 'cpa', 'roas'])

        # 🧮 Extract purchase value and count from nested fields
        # def extract_purchase(acts):
        #     total = 0.0
        #     if isinstance(acts, list):
        #         for a in acts:
        #             act_type = a.get("action_type", "").lower()
        #             if "purchase" in act_type:
        #                 try:
        #                     total += float(a.get("value", 0))
        #                 except:
        #                     continue
        #     return total
        
        def parse_actions_for_purchases(actions_list):
            purchases_count = 0
            purchase_value = 0.0
            if isinstance(actions_list, list):
                for action in actions_list:
                    if isinstance(action, dict):
                        action_type = action.get("action_type")
                        value = action.get("value")
                        if action_type == "offsite_conversion.fb_pixel_purchase": # Primary purchase action type
                            try:
                                purchases_count += float(value)
                            except (ValueError, TypeError):
                                pass
                        elif action_type == "purchase": # Another common purchase type
                             try:
                                purchases_count += float(value)
                             except (ValueError, TypeError):
                                pass
            return purchases_count


        # def extract_purchase_value(vals):
        #     total = 0.0
        #     if isinstance(vals, list):
        #         for a in vals:
        #             if isinstance(a, dict) and a.get("action_type") == "purchase":
        #                 try:
        #                     total += float(a.get("value", 0))
        #                 except:
        #                     continue
        #     return total
        
        def parse_action_values_for_purchase_value(action_values_list):
            total_purchase_value = 0.0
            if isinstance(action_values_list, list):
                for action_value in action_values_list:
                    if isinstance(action_value, dict):
                        action_type = action_value.get("action_type")
                        value = action_value.get("value")
                        if action_type == "offsite_conversion.fb_pixel_purchase": # Primary purchase value type
                            try:
                                total_purchase_value += float(value)
                            except (ValueError, TypeError):
                                pass
                        elif action_type == "purchase": # Another common purchase value type
                             try:
                                total_purchase_value += float(value)
                             except (ValueError, TypeError):
                                pass
            return total_purchase_value

        
        # print("🔍 Sample actions list:", df['actions'].iloc[0] if not df.empty else "No data")
        # print("🧪 Actions field sample:", json.dumps(df['actions'].tolist(), indent=2))


        # df['purchase_value'] = df['action_values'].apply(extract_purchase_value)
        # df['purchases'] = df['actions'].apply(extract_purchase)
        # print("🧪 Purchases extracted:", df['purchases'].describe())
        
        # Apply the new extraction functions
        df['purchases'] = df['actions'].apply(parse_actions_for_purchases)
        df['purchase_value'] = df['action_values'].apply(parse_action_values_for_purchase_value)

        # 🧹 Ensure all numeric fields exist and are cleaned


        # 🧹 Ensure all numeric fields exist and are cleaned
        required_numeric_cols = ['spend', 'impressions', 'clicks', 'reach', 'purchases', 'purchase_value']
        for col in required_numeric_cols:
            if col not in df.columns:
                df[col] = 0 # Add column if missing
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) # Convert to numeric and fill NaN

        # 🧠 Derived metrics, handling potential division by zero
        #df['cpa'] = df.apply(lambda row: row['spend'] / row['purchases'] if row['purchases'] > 0 else 0, axis=1) NA
        df['cpa'] = df.apply(lambda row: row['spend'] / row['purchases'] if row['purchases'] > 0 else np.nan, axis=1)
        df['roas'] = df.apply(lambda row: row['purchase_value'] / row['spend'] if row['spend'] > 0 else 0, axis=1)

        logger.info(f"📊 Demographic DataFrame Columns (after processing): {df.columns.tolist()}")
        logger.info(f"📊 Demographic DataFrame Preview (after processing): \n{df.head(2)}")
        logger.info(f"📊 Demographic DataFrame 'purchases' describe: \n{df['purchases'].describe().to_string()}")
        logger.info(f"📊 Demographic DataFrame 'purchase_value' describe: \n{df['purchase_value'].describe().to_string()}")


        # Return only the relevant columns
        return df[['age', 'gender', 'spend', 'purchases', 'purchase_value', 'cpa', 'roas']]
    
# services/deepseek_audit.py

import httpx
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
import json
import logging

logger = logging.getLogger(__name__)

DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# ... (other existing functions like generate_chart_1, generate_key_metrics_section, etc.)

async def fetch_platform_insights(account_id: str, user_token: str) -> pd.DataFrame:
    """
    Fetches platform-level insights from Facebook Graph API for a given ad account.
    """
    url = f"https://graph.facebook.com/v22.0/{account_id}/insights"
    now = datetime.now(timezone.utc)
    safe_until = (now - timedelta(days=2)).strftime("%Y-%m-%d")
    safe_since = (now - timedelta(days=32)).strftime("%Y-%m-%d")

    params = {
        "fields": "spend,impressions,clicks,reach,actions,action_values,date_start",
        "level": "ad", # Fetching at ad level to get publisher_platform breakdown
        "breakdowns": "publisher_platform",
        "time_range": json.dumps({"since": safe_since, "until": safe_until}),
        "time_increment": 1,
        "access_token": user_token
    }

    platform_data_raw = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            logger.info(f"Attempting to fetch platform insights for account {account_id} with params: {params}")
            response = await client.get(url, params=params)
            response.raise_for_status()
            data_page = response.json()
            logger.info(f"Raw API response data_page (first 5 items): {pprint.pformat(data_page.get('data', [])[:5])}")
            if "error" in data_page:
                logger.error(f"Facebook Graph API Error in platform insights: {data_page['error']['message']}")
                return pd.DataFrame() # Return empty if API returns an error
            platform_data_raw.extend(data_page.get("data", []))

            while data_page.get("paging", {}).get("next"):
                next_url = data_page["paging"]["next"]
                logger.info(f"Fetching next page for platform insights: {next_url}")
                next_response = await client.get(next_url, follow_redirects=True)
                next_response.raise_for_status()
                data_page = next_response.json()
                platform_data_raw.extend(data_page.get("data", []))
            
            logger.info(f"✅ Fetched {len(platform_data_raw)} platform insights for account {account_id}")

            if not platform_data_raw:
                logger.warning("No raw platform data found after fetching all pages.")
                return pd.DataFrame(columns=[
                    'date_start', 'spend', 'impressions', 'clicks', 'reach',
                    'purchases', 'purchase_value', 'publisher_platform'
                ])

            df = pd.DataFrame(platform_data_raw)
            
            logger.info(f"DataFrame created from raw data. Columns: {df.columns.tolist()}")
            logger.info(f"DataFrame head (before processing): \n{df.head().to_string()}")
            logger.info(f"Unique 'publisher_platform' values (before fillna): {df.get('publisher_platform', pd.Series(dtype='object')).unique()}")

            # Ensure all required columns exist and are numeric, filling NaNs
            required_numeric_cols = ['spend', 'impressions', 'clicks', 'reach']
            for col in required_numeric_cols:
                if col not in df.columns:
                    df[col] = 0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Handle actions and action_values for purchases and purchase_value
            PURCHASE_KEYS = [
                "offsite_conversion.purchase",
                "offsite_conversion.fb_pixel_purchase",
                "offsite_conversion.fb_pixel_custom",
                "offsite_conversion.custom.1408006162945363",
                "offsite_conversion.custom.587738624322885",
                "purchase"
            ]

            # def extract_purchase(acts):
            #     total = 0.0
            #     if isinstance(acts, list):
            #         for a in acts:
            #             act_type = a.get("action_type", "").lower()
            #             if "purchase" in act_type:
            #                 try:
            #                     total += float(a.get("value", 0))
            #                 except (ValueError, TypeError):
            #                     continue # Handle non-numeric values gracefully
            #     return total
            

            # def extract_purchase_value(vals):
            #     total = 0.0
            #     if isinstance(vals, list):
            #         for a in vals:
            #             if isinstance(a, dict) and a.get("action_type") == "purchase":
            #                 try:
            #                     total += float(a.get("value", 0))
            #                 except (ValueError, TypeError):
            #                     continue # Handle non-numeric values gracefully
            
            def extract_purchase(acts):
                total = 0.0
                if isinstance(acts, list):
                    for a in acts:
                        act_type = a.get("action_type", "").lower()
                        # Explicitly check for known purchase action types
                        if "purchase" in act_type or act_type == "offsite_conversion.fb_pixel_purchase":
                            try:
                                total += float(a.get("value", 0))
                            except (ValueError, TypeError):
                                continue
                return total

            def extract_purchase_value(vals):
                total = 0.0
                if isinstance(vals, list):
                    for a in vals:
                        if isinstance(a, dict):
                            act_type = a.get("action_type", "")
                            # Explicitly check for known purchase action types
                            if act_type == "purchase" or act_type == "offsite_conversion.fb_pixel_purchase":
                                try:
                                    total += float(a.get("value", 0))
                                except (ValueError, TypeError):
                                    continue
                return total
            
            # Check if 'actions' and 'action_values' columns exist before applying
            if 'actions' in df.columns:
                df['purchases'] = df['actions'].apply(extract_purchase)
            else:
                df['purchases'] = 0 # Default to 0 if column is missing
                logger.warning(" 'actions' column not found in raw platform data. 'purchases' set to 0.")

            if 'action_values' in df.columns:
                df['purchase_value'] = df['action_values'].apply(extract_purchase_value)
            else:
                df['purchase_value'] = 0 # Default to 0 if column is missing
                logger.warning(" 'action_values' column not found in raw platform data. 'purchase_value' set to 0.")

            # Ensure 'publisher_platform' exists and fill NaNs
            if 'publisher_platform' not in df.columns:
                logger.warning(" 'publisher_platform' column not found in DataFrame. Adding 'unknown'.")
                df['publisher_platform'] = 'unknown'
            else:
                df['publisher_platform'] = df['publisher_platform'].fillna('unknown')
                
            # Convert date_start to datetime for grouping
            if 'date_start' in df.columns:
                df['date_start'] = pd.to_datetime(df['date_start'], errors='coerce')
                df = df.dropna(subset=['date_start']) # Drop rows where date_start couldn't be parsed
            else:
                logger.error(" 'date_start' column is missing in platform insights data. Cannot group by date.")
                return pd.DataFrame()

            # Aggregate by platform and date to get daily platform data
            # This is important as raw insights can have multiple entries for platform/date combinations
            df_agg = df.groupby(['date_start', 'publisher_platform']).agg(
                spend=('spend', 'sum'),
                impressions=('impressions', 'sum'),
                clicks=('clicks', 'sum'),
                reach=('reach', 'sum'),
                purchases=('purchases', 'sum'),
                purchase_value=('purchase_value', 'sum')
            ).reset_index()

            # Rename 'publisher_platform' to 'platform' for consistency with downstream functions
            df_agg.rename(columns={'publisher_platform': 'platform'}, inplace=True)
            # --- Final DataFrame Debugging ---
            logger.info(f"📊 Platform DataFrame (aggregated and renamed). Columns: {df_agg.columns.tolist()}")
            logger.info(f"📊 Platform DataFrame (aggregated and renamed) Head:\n{df_agg.head().to_string()}")
            logger.info(f"Unique 'platform' values in aggregated DF: {df_agg['platform'].unique()}")

            
            logger.info(f"📊 Platform DataFrame (aggregated) Head:\n{df_agg.head()}")
            return df_agg

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching platform insights: {e.response.status_code} - {e.response.text} for URL: {e.request.url}")
            logger.error(f"Response headers: {e.response.headers}")
            logger.error(f"Response content: {e.response.text}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching platform insights: {str(e)}")
            return pd.DataFrame()

# services/deepseek_audit.py

# ... (other existing functions)

def group_by_platform(df: pd.DataFrame, currency_symbol="₹") -> pd.DataFrame:
    """
    Groups the provided DataFrame by 'platform' and calculates aggregated metrics.
    Assumes 'platform' column already exists in the DataFrame.
    """
    df_copy = df.copy() # Use a copy to avoid SettingWithCopyWarning

    # Ensure numeric columns, handling potential missing columns safely
    for col in ['spend', 'purchase_value', 'purchases']:
        if col not in df_copy.columns:
            df_copy[col] = 0 # Add as 0 if missing
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)

    # Ensure 'platform' column exists and fill NaN values
    if 'platform' not in df_copy.columns:
        logger.warning("No 'platform' column found in the DataFrame passed to group_by_platform. Adding 'Unknown'.")
        df_copy['platform'] = 'Unknown'
    else:
        df_copy['platform'] = df_copy['platform'].fillna("Unknown")

    # Calculate ROAS and CPA, handling division by zero
    df_copy['roas'] = df_copy['purchase_value'] / df_copy['spend'].replace(0, 1)
    #df_copy['cpa'] = df_copy['spend'] / df_copy['purchases'].replace(0, 1) NA
    # Ensure purchases column is numeric first
    df_copy['purchases'] = pd.to_numeric(df_copy['purchases'], errors='coerce').fillna(0)
    # Calculate CPA, setting to NaN if purchases are 0
    df_copy['cpa'] = df_copy.apply(lambda row: row['spend'] / row['purchases'] if row['purchases'] > 0 else np.nan, axis=1)

    grouped = df_copy.groupby('platform').agg(
        spend=('spend', 'sum'),
        purchase_value=('purchase_value', 'sum'),
        purchases=('purchases', 'sum'),
        roas=('roas', 'mean'), # Using mean for aggregated ROAS
        cpa=('cpa', 'mean')   # Using mean for aggregated CPA
    ).reset_index()

    grouped['spend'] = grouped['spend'].round(2)
    grouped['purchase_value'] = grouped['purchase_value'].round(2)
    grouped['roas'] = grouped['roas'].round(2)
    grouped['cpa'] = grouped['cpa'].round(2)

    return grouped

# async def generate_platform_summary(df, currency_symbol):
#     df = df.copy()
#     df['roas'] = df['purchase_value'] / df['spend'].replace(0, 1)
#     df['cpa'] = df['spend'] / df['purchases'].replace(0, 1)
#     df = df.groupby('platform').agg({
#         'spend': 'sum',
#         'purchase_value': 'sum',
#         'purchases': 'sum',
#         'roas': 'mean',
#         'cpa': 'mean'
#     }).reset_index()

#     summary_data = df.to_dict(orient='records')
#     prompt = f"""
#     Write a concise summary of platform-level performance across Meta Ads.
#     Include top-performing platforms (high ROAS or low CPA), and platforms underperforming.
#     Conclude with 1 actionable recommendation.
#     Use {currency_symbol} in monetary values.
#     """
#     return await generate_llm_content(prompt, summary_data)

async def generate_platform_summary(platform_df: pd.DataFrame, currency_symbol: str) -> str:
    """
    Generate a clean summary of platform-level ad performance.
    """
    if platform_df.empty or "Platform" not in platform_df.columns:
        return "⚠️ No platform performance data available."

    df = platform_df.copy()
    df.columns = [col.lower().strip().replace(" ", "_") for col in df.columns]

    df['amount_spent'] = pd.to_numeric(df['amount_spent'], errors='coerce').fillna(0)
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)
    df['purchases'] = pd.to_numeric(df['purchases'], errors='coerce').fillna(0)
    df['roas'] = df['revenue'] / df['amount_spent'].replace(0, 1)
    df['cpa'] = df['amount_spent'] / df['purchases'].replace(0, 1)

    platform_summaries = []
    for _, row in df.iterrows():
        platform = row['platform'].capitalize()
        spend = f"{currency_symbol}{row['amount_spent']:,.0f}"
        roas = f"{row['roas']:.2f}"
        purchases = int(row['purchases'])
        cpa = f"{currency_symbol}{row['cpa']:,.2f}" if row['purchases'] > 0 else "N/A"
        platform_summaries.append(f"{platform}: Spend = {spend}, Purchases = {purchases}, ROAS = {roas}, CPA = {cpa}")

    if df['purchases'].sum() == 0:
        recommendation = (
            "Pause all campaigns immediately and evaluate your ad targeting, creative performance, "
            "and landing page experience. Consider small-scale A/B tests to identify what drives conversions "
            "before investing further."
        )
    else:
        recommendation = (
            "Continue optimizing by allocating more budget to platforms with higher ROAS and lower CPA. "
            "Explore underperforming platforms to identify bottlenecks in the conversion funnel."
        )

    summary_data = (
        "### Meta Ads Platform Performance Summary\n\n"
        + "\n".join(platform_summaries) + "\n\n"
        + "**Recommendation:** " + recommendation
    )
    prompt = f"""
#     Write a concise summary of platform-level performance across Meta Ads.
#     Include top-performing platforms (high ROAS or low CPA), and platforms underperforming.
#     Conclude with 1 actionable recommendation.
#     Use {currency_symbol} in monetary values.
#     """

    return await generate_llm_content(prompt, summary_data)


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
                return [], pd.DataFrame() # Return empty list and empty demographic_df

            insights_data = []
            all_demographic_dfs = [] # To collect demographic data from all accounts

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
                
                # We will fetch demographic data using the dedicated fetch_demographic_insights function later
                # so no need for demographic_params here.

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
                    # This section correctly fetches reach_df
                    reach_url = f"https://graph.facebook.com/v22.0/{acc['id']}/insights"
                    reach_params = {
                        "fields": "adset_id,reach,date_start",
                        "time_range": json.dumps({"since": safe_since, "until": safe_until}),
                        "time_increment": 1,
                        "level": "adset",
                        "access_token": user_token
                    }
                    
                    reach_df = pd.DataFrame()
                    try:
                        reach_response = await client.get(reach_url, params=reach_params)
                        reach_response.raise_for_status()
                        reach_data = reach_response.json().get("data", [])
                        reach_df = pd.DataFrame(reach_data)
                    except Exception as e:
                        print(f"⚠️ Failed to fetch reach data for account {acc['id']}: {e}")
                        
                    # Demographic data will be fetched by generate_audit directly using fetch_demographic_insights
                    # No need for demographic_df handling here.
 
                        
                    # ✅ DEBUG: Print full data sample after all pages
                    print("📦 Final sample of fetched ad data (first 3 rows):")
                    pprint.pprint(ad_results[:3], indent=2)

                print(f"✅ Total insights for account {acc['id']}: {len(ad_results)}")
                
                # 🧠 Merge reach into ad-level data if available
                if not reach_df.empty:
                    reach_df["date_start"] = pd.to_datetime(reach_df["date_start"])
                    for ad in ad_results:
                        if "adset_id" in ad and "date_start" in ad:
                            # Ensure ad['date_start'] is also datetime for consistent comparison--newline
                            ad_date_start_dt = pd.to_datetime(ad["date_start"])
                            match = reach_df[
                                (reach_df["adset_id"] == ad["adset_id"]) &
                                (reach_df["date_start"] == ad_date_start_dt)
                                #newline
                                # (reach_df["adset_id"] == ad["adset_id"]) &
                                # (pd.to_datetime(reach_df["date_start"]) == pd.to_datetime(ad["date_start"]))
                            ]
                            if not match.empty:
                                ad["reach"] = match["reach"].values[0]
                                logger.debug(f"Merged reach {ad['reach']} for adset {ad['adset_id']} on {ad['date_start']}") # Added debug log newline
                            else:
                                logger.debug(f"No matching reach found for adset {ad['adset_id']} on {ad['date_start']}") # Added debug log newline
                                
                else:
                    logger.info("Reach DataFrame is empty, skipping reach merge.") # Added logger newline

                
                # Removed the problematic demographic_df = pd.DataFrame() initializations here
                # and the subsequent if not demographic_df.empty blocks.
                # demographic_df will now be fetched and handled at the generate_audit level.

                for ad in ad_results:
                    if 'account_currency' not in ad:
                        ad["account_currency"] = acc.get("account_currency", "USD")
                    ad["account_id"] = acc.get("id")
                    insights_data.append(ad)
                    
            logger.info(f"📦 Fetched total {len(insights_data)} ads across all accounts after all processing.") # Changed from print to logger.info newline
            
                    
            #print(f"📦 Fetched total {len(insights_data)} ads across all accounts.")
            
            # fetch_ad_insights should now only return insights_data, not demographic_df
            # demographic_df will be fetched separately in generate_audit
            return insights_data, safe_since, safe_until

    except Exception as e:
        print(f"❌ Error in fetch_ad_insights: {str(e)}")
        # Return empty list and empty DataFrame if an error occurs
        return [], pd.DataFrame()

# --- (LLM Truncation and Generation Functions - No changes needed here, keeping for context) ---

MAX_TOKEN_LIMIT = 8000
MAX_AD_ITEMS = 30

def truncate_ad_data(data: dict, max_items: int = MAX_AD_ITEMS) -> dict:
    truncated = data.copy()
    if "ad_insights" in truncated and isinstance(truncated["ad_insights"], list):
        truncated["ad_insights"] = truncated["ad_insights"][:max_items]
    return truncated

async def generate_llm_content(prompt: str, data: dict) -> str:
    try:
        truncated_data = truncate_ad_data(data)
        try:
            data_str = json.dumps(truncated_data, indent=2)
        except Exception as e:
            print(f"⚠️ Error serializing data to JSON: {e}")
            data_str = str(truncated_data)
        if len(data_str) > MAX_TOKEN_LIMIT:
            print(f"⚠️ Even truncated data too large ({len(data_str)} chars). Truncating...")
            data_str = data_str[:MAX_TOKEN_LIMIT] + "\n\n...[truncated]"
        user_prompt = f"{prompt}\n\nAnalyze the following Facebook data:\n{data_str}"
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a Meta Ads expert who writes audit reports."},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7
        }
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

# --- (generate_audit function - Most important changes here) ---

async def generate_audit(page_id: str, user_token: str, page_token: str):
    from services.generate_pdf import generate_pdf_report # Ensure this import is here
    
    """Generate audit report and return PDF"""
    try:
        print("🔄 Starting audit generation...")

        if not DEEPSEEK_API_URL or not DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_URL and DEEPSEEK_API_KEY environment variables must be set")

        print("📊 Fetching Facebook data...")
        page_data = await fetch_facebook_insights(page_id, page_token)
        
        # fetch_ad_insights now only returns ad_data (list of dicts)
        ad_data, date_since, date_until = await fetch_ad_insights(user_token) 
        
        print("🔍 ad_data structure after fetch_ad_insights:", type(ad_data))
        print("🔍 Raw ad data preview after fetch_ad_insights:", ad_data[:2])

        # Extract account_id for demographic insights
        account_id = None
        if ad_data and isinstance(ad_data, list) and isinstance(ad_data[0], dict):
            account_id = ad_data[0].get('account_id')
        elif isinstance(ad_data, list): # Iterate if first item not a dict
            for item in ad_data:
                if isinstance(item, dict) and 'account_id' in item:
                    account_id = item['account_id']
                    break
        print(f"🆔 Extracted Account ID: {account_id}")
        
        # --- CRITICAL FIX: Fetch demographic_df here, ONCE, and store it ---
        demographic_df = pd.DataFrame() # Initialize to empty before attempting to fetch
        if account_id:
            demographic_df = await fetch_demographic_insights(account_id, user_token)
            print(f"✅ Fetched demographic_df from fetch_demographic_insights. Shape: {demographic_df.shape}")
            print("📊 Demographic DataFrame Head (after fetch):", demographic_df.head())
        else:
            print("⚠️ Could not determine account_id. Skipping demographic insights fetch.")
            
        # --- NEW: Fetch platform data ---
        platform_df_raw = pd.DataFrame()
        if account_id:
            platform_df_raw = await fetch_platform_insights(account_id, user_token)
            print(f"✅ Fetched platform_df_raw. Shape: {platform_df_raw.shape}")
            print("📊 Platform DataFrame Raw Head:", platform_df_raw.head())
        else:
            print("⚠️ Could not determine account_id. Skipping platform insights fetch.")
        # --- End NEW ---

        
        # --- Removed the problematic demographic_df re-initialization and processing from here ---
        # The processing of demographic_df columns (spend, impressions, reach) and grouping
        # should happen within generate_pdf_report or fetch_demographic_insights, not here.
        # fetch_demographic_insights already handles this.

        # Filter out invalid entries from ad_data
        ad_data = [d for d in ad_data if isinstance(d, dict) and 'date_start' in d and d.get('date_start')]
        
        if not ad_data:
            print("❌ No usable ad entries with 'date_start' after initial fetch and filter. This might lead to an empty report.")
            # Depending on desired behavior, you might want to raise an error here
            # raise ValueError("No usable ad insights data available.")
        
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
                print(f"⚠️ Error parsing actions/values in ad: {e}")
                actions, values = {}, {}
            ad["purchases"] = sum(actions.get(k, 0) for k in PURCHASE_KEYS)
            
            if 'reach' not in ad:
                if 'impressions' in ad:
                    ad["reach"] = ad["impressions"]
                else:
                    ad["reach"] = 1 # Fallback to avoid division errors

            raw_value = sum(values.get(k, 0) for k in PURCHASE_KEYS)
            if raw_value == 0 and ad["purchases"] > 0:
                raw_value = ad["purchases"] * 1000 # Assume 1000 per purchase as fallback
                ad["purchase_value"] = raw_value
            else:
                ad["purchase_value"] = raw_value # Ensure purchase_value is set even if raw_value is 0

            ad["link_clicks"] = actions.get("link_click", 0)
            
            if "purchase_value" not in ad or not isinstance(ad["purchase_value"], (int, float)):
                ad["purchase_value"] = 0
            if "purchases" not in ad or not isinstance(ad["purchases"], (int, float)):
                ad["purchases"] = 0

        # Create original DataFrame with date_start intact
        original_df = pd.DataFrame(ad_data)
        
        # --- Logger: Check 'reach' column in original_df after its creation --- newline
        logger.info(f"📊 original_df columns after initial processing: {original_df.columns.tolist()}") # Added logger
        logger.info(f"📊 original_df head (with reach) before final date processing: \n{original_df[['ad_name', 'impressions', 'reach']].head().to_string()}") # Added logger
        logger.info(f"📊 original_df descriptive stats for 'reach': \n{original_df['reach'].describe().to_string()}") # Added logger

        # Ensure reach column exists
        if 'reach' not in original_df.columns:
            if 'impressions' in original_df.columns:
                print("⚠️ 'reach' column missing in original_df, using impressions as fallback")
                original_df['reach'] = original_df['impressions']
            else:
                print("⚠️ Both 'reach' and 'impressions' missing in original_df - setting reach to 1")
                original_df['reach'] = 1
        
        original_df['campaign_name'] = original_df['campaign_name'].fillna("Unknown Campaign")
        original_df['adset_name'] = original_df['adset_name'].fillna("Unknown Adset")

        if 'account_currency' not in original_df.columns:
            print("⚠️ 'account_currency' column missing in original_df. Defaulting to USD.")
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
            'cpc', 'ctr', 'reach', 'link_clicks' # Added reach and link_clicks for consistency
        ]
        for col in numeric_fields:
            if col in original_df.columns:
                original_df[col] = pd.to_numeric(original_df[col], errors='coerce').fillna(0)
                
        # Handle 'conversion_value' separately if it's not purchase_value
        if 'conversion_value' in original_df.columns:
             original_df['conversion_value'] = pd.to_numeric(original_df['conversion_value'], errors='coerce').fillna(0)
        else:
             original_df['conversion_value'] = original_df['purchase_value'] # Fallback if not provided

        # Handle 'conversions' separately if it's not purchases
        if 'conversions' in original_df.columns:
            original_df['conversions'] = pd.to_numeric(original_df['conversions'], errors='coerce').fillna(0)
        else:
            original_df['conversions'] = original_df['purchases'] # Fallback if not provided

        # Calculate aggregated metrics per day
        grouped_df = original_df.groupby('date').agg({
            'spend': 'sum',
            'purchases': 'sum',
            'purchase_value': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'cpc': 'mean', # Mean of daily CPCs
            'ctr': 'mean', # Mean of daily CTRs
            'reach': 'sum', # Sum reach per day for frequency calculation
            'link_clicks': 'sum' # Sum link clicks per day
        }).reset_index()

        # Recalculate daily level metrics after aggregation
        grouped_df['roas'] = grouped_df['purchase_value'] / grouped_df['spend'].replace(0, 1)
        #grouped_df['cpa'] = grouped_df['spend'] / grouped_df['purchases'].replace(0, 1) NA
        # Ensure purchases are numeric and handle NaN
        grouped_df['purchases'] = pd.to_numeric(grouped_df['purchases'], errors='coerce').fillna(0)
        grouped_df['cpa'] = grouped_df.apply(lambda row: row['spend'] / row['purchases'] if row['purchases'] > 0 else np.nan, axis=1)
        grouped_df['click_to_conversion'] = grouped_df['purchases'] / grouped_df['clicks'].replace(0, 1)
        grouped_df['frequency'] = grouped_df['impressions'] / grouped_df['reach'].replace(0,1) # Calculate daily frequency

        cutoff = pd.Timestamp.today() - pd.Timedelta(days=30)
        ad_insights_df = grouped_df[grouped_df['date'] >= cutoff].copy()

        if ad_insights_df.empty:
            print("⚠️ No data in last 30 days. Using last available 30 records from grouped_df.")
            ad_insights_df = grouped_df.tail(30).copy() # Use .copy() to avoid SettingWithCopyWarning
        
        # Ensure ad_insights_df has expected columns after potential tail() operation
        for col in ['cpc', 'ctr', 'roas', 'cpa', 'click_to_conversion', 'frequency']:
            if col not in ad_insights_df.columns:
                ad_insights_df[col] = 0 # Fallback default
        
        currency = "USD"
        currency_symbol = "$"

        def detect_currency(df):
            if 'account_currency' not in df.columns:
                print("⚠️ No 'account_currency' column found in the DataFrame")
                return "USD", "$"
            
            currency_symbols = {
                "INR": "₹",
                "USD": "$",
            }
            
            currencies = df['account_currency'].dropna().astype(str).str.strip().str.upper()
            if currencies.empty:
                print("⚠️ No valid currency values found in 'account_currency' column")
                return "USD", "$"
            
            unique_currencies = currencies.unique()
            print(f"🔍 Unique currency values found: {unique_currencies}")
            
            currency = currencies.mode()[0] if not currencies.mode().empty else "USD"
            currency_symbol = currency_symbols.get(currency, "$")
            
            print(f"✅ Using currency: {currency} with symbol: {currency_symbol}")
            return currency, currency_symbol

        currency, currency_symbol = detect_currency(original_df)
        print(f"💰 Detected account currency: {currency} → Using symbol: {currency_symbol}")

        combined_data = {
            "page_insights": page_data,
            "ad_insights": ad_data # This is the raw ad_data list, not the DataFrame
        }

        key_metrics = generate_key_metrics_section(ad_insights_df, currency_symbol=currency_symbol)
        split_charts = generate_campaign_split_charts(original_df, currency_symbol)
        
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

        sections = [
            {"title": "EXECUTIVE SUMMARY", "content": executive_summary, "charts": []},
            {"title": "ACCOUNT NAMING & STRUCTURE", "content": account_structure, "charts": []},
            {"title": "TESTING ACTIVITY", "content": testing_activity, "charts": []},
            {"title": "REMARKETING ACTIVITY", "content": remarketing_activity, "charts": []},
            {"title": "RESULTS SETUP", "content": results_setup, "charts": []},
            key_metrics
        ]
        
        # Add a placeholder for Demographic Performance
        sections.append({"title": "DEMOGRAPHIC PERFORMANCE", "content": "", "charts": []})
        sections.append({"title": "Platform Level Performance","contains_table": True})



        print("📊 Sample of original_df:")
        print(original_df[["date", "campaign_name", "spend", "purchase_value", "purchases"]].tail(5))

        print("📄 Generating PDF report...")
        pdf_response = generate_pdf_report(
            sections,
            ad_insights_df=ad_insights_df,
            full_ad_insights_df=original_df,
            currency_symbol=currency_symbol,
            split_charts=split_charts,
            demographic_df=demographic_df,
            platform_df=platform_df_raw,
            date_since=date_since, 
            date_until=date_until
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