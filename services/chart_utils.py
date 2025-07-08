import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import BytesIO
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates

from services.prompts import (
    EXECUTIVE_SUMMARY_PROMPT,
    ACCOUNT_NAMING_STRUCTURE_PROMPT,
    TESTING_ACTIVITY_PROMPT,
    REMARKETING_ACTIVITY_PROMPT,
    RESULTS_SETUP_PROMPT
)


def generate_chart_image(fig):
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=200)  # Removed bbox_inches
    buf.seek(0)
    plt.close(fig)
    return buf

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

import matplotlib.pyplot as plt
from io import BytesIO

def generate_bar_chart(series, title, color="#1f77b4"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(series.index[::-1], series.values[::-1], color=color)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                f"{width:.2f}", va='center', fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("Amount")
    ax.set_xlim(left=0)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()

    # Convert to image bytes
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    return (title, buffer)

def generate_cost_by_adset_chart(df):
    grouped = df.copy()
    grouped = grouped[grouped['adset_name'].notna()]
    grouped = grouped.groupby('adset_name')['spend'].sum().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    bars = ax.barh(grouped.index[::-1], grouped.values[::-1], color="#4E79A7", height=0.5)

    # Add value labels to bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (0.02 * width), bar.get_y() + bar.get_height()/2,
                f"{width:,.2f}", va='center', fontsize=9, color="#333333")

    # Modern UI tweaks
    ax.set_title("Cost by Adsets", fontsize=14, fontweight="bold", color="#333333")
    ax.set_xlabel("Amount Spent", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    return ("Cost by Adsets", generate_chart_image(fig))


def generate_revenue_by_adset_chart(df):
    grouped = df.copy()
    grouped = grouped[grouped['adset_name'].notna()]
    grouped = grouped.groupby('adset_name')['purchase_value'].sum().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    bars = ax.barh(grouped.index[::-1], grouped.values[::-1], color="#F28E2B", height=0.5)

    # Add value labels to bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (0.02 * width), bar.get_y() + bar.get_height()/2,
                f"{width:,.2f}", va='center', fontsize=9, color="#333333")

    # Modern UI tweaks
    ax.set_title("Revenue by Adsets", fontsize=14, fontweight="bold", color="#333333")
    ax.set_xlabel("Revenue", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    return ("Revenue by Adsets", generate_chart_image(fig))

