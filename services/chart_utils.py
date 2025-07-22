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
    fig, ax = plt.subplots(figsize=(7, 4))  # Optional: slightly wider chart
    bars = ax.barh(roas_series.index, roas_series.values, color="#007fff", height=0.4)

    ax.set_xlabel("ROAS")
    ax.set_title("ROAS Split by Adset")

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}", va='center', fontsize=8)

    # 🔧 Make bars longer by increasing x-axis range
    max_val = roas_series.max()
    ax.set_xlim(0, max_val * 2)  # ← increases available space to stretch bars

    # Clean up axes
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
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['spend'] = pd.to_numeric(df['spend'], errors='coerce').fillna(0)
    df['adset_name'] = df['adset_name'].fillna("Unknown Adset")
    
    grouped = df.groupby(['adset_name', 'date'])['spend'].sum().reset_index()
    pivot_df = grouped.pivot(index='date', columns='adset_name', values='spend').fillna(0)

    fig, ax = plt.subplots(figsize=(14, 8), dpi=200)

    color_cycle = plt.cm.tab10.colors  # or use plt.get_cmap("tab20").colors
    lines = []
    for i, column in enumerate(pivot_df.columns):
        line, = ax.plot(pivot_df.index, pivot_df[column], label=column,
                        linewidth=2, marker='o' if i < 2 else '', markersize=4,
                        color=color_cycle[i % len(color_cycle)])
        lines.append(line)

    ax.set_title("Cost By Adsets", fontsize=16, weight='bold')
    ax.set_ylabel("Amount Spent")
    ax.set_xlabel("Day")

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.3)

    ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),  # ✅ move legend slightly *inside* the plot area
    ncol=3,
    fontsize=8,
    frameon=False
    )


    fig.tight_layout()
    return ("Cost by Adsets", generate_chart_image(fig))


def generate_revenue_by_adset_chart(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['purchase_value'] = pd.to_numeric(df['purchase_value'], errors='coerce').fillna(0)
    df['adset_name'] = df['adset_name'].fillna("Unknown Adset")

    grouped = df.groupby(['adset_name', 'date'])['purchase_value'].sum().reset_index()
    pivot_df = grouped.pivot(index='date', columns='adset_name', values='purchase_value').fillna(0)

    fig, ax = plt.subplots(figsize=(14, 8), dpi=200)

    color_cycle = plt.cm.tab10.colors
    for i, column in enumerate(pivot_df.columns):
        ax.plot(pivot_df.index, pivot_df[column], label=column,
                linewidth=2, marker='o' if i < 2 else '', markersize=4,
                color=color_cycle[i % len(color_cycle)])

    ax.set_title("Revenue By Adsets", fontsize=16, weight='bold')
    ax.set_ylabel("Revenue")
    ax.set_xlabel("Day")

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.3)

    ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),  # ✅ move legend slightly *inside* the plot area
    ncol=3,
    fontsize=8,
    frameon=False
    )

    fig.tight_layout()
    return ("Revenue by Adsets", generate_chart_image(fig))


def generate_frequency_over_time_chart(df):
    df['date'] = pd.to_datetime(df['date'])
    df['frequency'] = df['impressions'] / df['reach'].replace(0, 1)
    pivot_df = df.pivot_table(index='date', columns='ad_name', values='frequency').fillna(0)

    fig, ax = plt.subplots(figsize=(15, 7), dpi=200)
    for column in pivot_df.columns[:5]:
        ax.plot(pivot_df.index, pivot_df[column], label=column, linewidth=1.5)
    ax.axhline(y=3, color='red', linestyle='--', linewidth=1, label="Fatigue Threshold (3)")
    ax.set_title("Frequency Over Time", fontsize=16, weight='bold')
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Day")
    ax.legend(loc="upper center", fontsize=8, ncol=3)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    return ("Frequency Over Time", generate_chart_image(fig))


def generate_cpm_over_time_chart(df):
    df['date'] = pd.to_datetime(df['date'])
    df['cpm'] = (df['spend'] / df['impressions'].replace(0, 1)) * 1000
    pivot_df = df.pivot_table(index='date', columns='ad_name', values='cpm').fillna(0)

    fig, ax = plt.subplots(figsize=(15, 7), dpi=200)
    for column in pivot_df.columns[:5]:
        ax.plot(pivot_df.index, pivot_df[column], label=column, linewidth=1.5)
    ax.set_title("CPM Over Time", fontsize=16, weight='bold')
    ax.set_ylabel("CPM (₹)")
    ax.set_xlabel("Day")
    ax.legend(loc="upper center", fontsize=8, ncol=3)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    return ("CPM Over Time", generate_chart_image(fig))

# def generate_cost_split_by_age_chart(df):
#     if 'Age' not in df.columns or 'Amount Spent' not in df.columns:
#         raise ValueError("Required columns 'Age' and 'Amount Spent' not found")
#     grouped = df.groupby('Age')['Amount Spent'].sum()
#     fig, ax = plt.subplots(figsize=(5, 5))
#     ax.pie(grouped, labels=grouped.index, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4))
#     ax.set_title("Cost Split By Age", fontsize=14)
#     plt.tight_layout()
#     buf = BytesIO()
#     fig.savefig(buf, format='png', dpi=200)
#     buf.seek(0)
#     plt.close(fig)
#     return buf
# def generate_revenue_split_by_age_chart(df):
#     if 'Age' not in df.columns or 'Purchases' not in df.columns:
#         raise ValueError("Required columns 'Age' and 'Amount Spent' not found")
#     grouped = df.groupby('Age')['Purchases'].sum()
#     fig, ax = plt.subplots(figsize=(5, 5))
#     ax.pie(grouped, labels=grouped.index, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4))
#     ax.set_title("Revenue Split By Age", fontsize=14)
#     plt.tight_layout()
#     buf = BytesIO()
#     fig.savefig(buf, format='png', dpi=200)
#     buf.seek(0)
#     plt.close(fig)
#     return buf
# def generate_cost_split_by_gender_chart(df):
#     if 'Gender' not in df.columns or 'Amount Spent' not in df.columns:
#         raise ValueError("Required columns 'Age' and 'Amount Spent' not found")
#     grouped = df.groupby('Gender')['Amount Spent'].sum()
#     fig, ax = plt.subplots(figsize=(5, 5))
#     ax.pie(grouped, labels=grouped.index, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4))
#     ax.set_title("Cost Split By Gender", fontsize=14)
#     plt.tight_layout()
#     buf = BytesIO()
#     fig.savefig(buf, format='png', dpi=200)
#     buf.seek(0)
#     plt.close(fig)
#     return buf
# def generate_revenue_split_by_gender_chart(df):
#     if 'Gender' not in df.columns or 'Purchases' not in df.columns:
#         raise ValueError("Required columns 'Age' and 'Amount Spent' not found")
#     grouped = df.groupby('Gender')['Purchases'].sum()
#     fig, ax = plt.subplots(figsize=(5, 5))
#     ax.pie(grouped, labels=grouped.index, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4))
#     ax.set_title("Revenue Split By Gender", fontsize=14)
#     plt.tight_layout()
#     buf = BytesIO()
#     fig.savefig(buf, format='png', dpi=200)
#     buf.seek(0)
#     plt.close(fig)
#     return buf
# def generate_roas_split_by_age_chart(df):
#     grouped = df.groupby('Age')['ROAS'].mean()
#     fig, ax = plt.subplots(figsize=(6, 4))
#     bars = ax.barh(grouped.index, grouped.values, color="#ff00aa")
#     ax.set_title("ROAS Split By Age", fontsize=14)
#     ax.set_xlabel("ROAS")
#     for bar in bars:
#         width = bar.get_width()
#         ax.text(width + 0.05, bar.get_y() + bar.get_height() / 2, f"{width:.2f}", va='center', fontsize=8)
#     plt.tight_layout()
#     buf = BytesIO()
#     fig.savefig(buf, format='png', dpi=200)
#     buf.seek(0)
#     plt.close(fig)
#     return buf
# def generate_roas_split_by_gender_chart(df):
#     grouped = df.groupby('Gender')['ROAS'].mean()
#     fig, ax = plt.subplots(figsize=(6, 4))
#     bars = ax.barh(grouped.index, grouped.values, color="#ff00aa")
#     ax.set_title("ROAS Split By Gender", fontsize=14)
#     ax.set_xlabel("ROAS")
#     for bar in bars:
#         width = bar.get_width()
#         ax.text(width + 0.05, bar.get_y() + bar.get_height() / 2, f"{width:.2f}", va='center', fontsize=8)
#     plt.tight_layout()
#     buf = BytesIO()
#     fig.savefig(buf, format='png', dpi=200)
#     buf.seek(0)
#     plt.close(fig)
#     return buf

from io import BytesIO
import matplotlib.pyplot as plt

# Pie chart colors
PIE_COLORS = ['#4CAF50', '#2196F3', '#FFC107', '#FF5722', '#9C27B0', '#00BCD4']

# def generate_cost_split_by_age_chart(df):
#     if 'Age' not in df.columns or 'Amount Spent' not in df.columns:
#         raise ValueError("Required columns 'Age' and 'Amount Spent' not found")
#     grouped = df.groupby('Age')['Amount Spent'].sum()
#     fig, ax = plt.subplots(figsize=(5, 5))
#     ax.pie(grouped, labels=grouped.index, autopct='%1.1f%%', startangle=90,
#            wedgeprops=dict(width=0.4), colors=PIE_COLORS)
#     ax.set_title("Cost Split By Age", fontsize=14)
#     plt.tight_layout()
#     buf = BytesIO()
#     fig.savefig(buf, format='png', dpi=200)
#     buf.seek(0)
#     plt.close(fig)
#     return buf

def generate_cost_split_by_age_chart(df):
    # Ensure column name consistency
    df = df.rename(columns={'Amount Spent': 'amount_spent', 'Age': 'age'})
    if 'age' not in df.columns or 'amount_spent' not in df.columns:
        raise ValueError("Required columns not found")
    
    grouped = df.groupby('age')['amount_spent'].sum()
    if grouped.sum() <= 0:
        return create_empty_chart_image("No spend data by age")
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(grouped, labels=grouped.index, autopct='%1.1f%%', 
           startangle=90, wedgeprops=dict(width=0.4), colors=PIE_COLORS)
    ax.set_title("Cost Split By Age", fontsize=14)
    plt.tight_layout()
    return generate_chart_image(fig)

def create_empty_chart_image(message):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.text(0.5, 0.5, message, 
            ha='center', va='center', 
            fontsize=12, color='red')
    ax.axis('off')
    return generate_chart_image(fig)

def generate_revenue_split_by_age_chart(df):
    # Standardize column names
    df = df.rename(columns={
        'Age': 'age',
        'Purchases': 'purchases'
    })
    
    if 'age' not in df.columns or 'purchases' not in df.columns:
        raise ValueError("Required columns 'age' and 'purchases' not found")
    
    # Filter out rows with zero purchases
    df = df[df['purchases'] > 0]
    
    if df.empty:
        return create_empty_chart_image("No purchase data by age")
    
    grouped = df.groupby('age')['purchases'].sum()
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(grouped, labels=grouped.index, autopct='%1.1f%%', 
           startangle=90, wedgeprops=dict(width=0.4), colors=PIE_COLORS)
    ax.set_title("Revenue Split By Age", fontsize=14)
    plt.tight_layout()
    return generate_chart_image(fig)

def generate_cost_split_by_gender_chart(df):
    # Standardize column names
    df = df.rename(columns={
        'Gender': 'gender',
        'Amount Spent': 'amount_spent'
    })
    
    if 'gender' not in df.columns or 'amount_spent' not in df.columns:
        raise ValueError("Required columns 'gender' and 'amount_spent' not found")
    
    # Filter out rows with zero or negative spend
    df = df[df['amount_spent'] > 0]
    
    if df.empty:
        return create_empty_chart_image("No spend data by gender")
    
    grouped = df.groupby('gender')['amount_spent'].sum()
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(grouped, labels=grouped.index, autopct='%1.1f%%', 
           startangle=90, wedgeprops=dict(width=0.4), colors=PIE_COLORS)
    ax.set_title("Cost Split By Gender", fontsize=14)
    plt.tight_layout()
    return generate_chart_image(fig)

def generate_revenue_split_by_gender_chart(df):
    # Standardize column names
    df = df.rename(columns={
        'Gender': 'gender',
        'Purchases': 'purchases'
    })
    
    if 'gender' not in df.columns or 'purchases' not in df.columns:
        raise ValueError("Required columns 'gender' and 'purchases' not found")
    
    # Filter out rows with zero purchases
    df = df[df['purchases'] > 0]
    
    if df.empty:
        return create_empty_chart_image("No purchase data by gender")
    
    grouped = df.groupby('gender')['purchases'].sum()
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(grouped, labels=grouped.index, autopct='%1.1f%%', 
           startangle=90, wedgeprops=dict(width=0.4), colors=PIE_COLORS)
    ax.set_title("Revenue Split By Gender", fontsize=14)
    plt.tight_layout()
    return generate_chart_image(fig)

def generate_roas_split_by_age_chart(df):
    # Standardize column names
    df = df.rename(columns={
        'Age': 'age',
        'ROAS': 'roas'
    })
    
    if 'age' not in df.columns or 'roas' not in df.columns:
        raise ValueError("Required columns 'age' and 'roas' not found")
    
    # Filter out invalid ROAS values
    df = df[df['roas'].notna() & (df['roas'] >= 0)]
    
    if df.empty:
        return create_empty_chart_image("No valid ROAS data by age")
    
    grouped = df.groupby('age')['roas'].mean()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(grouped.index, grouped.values, color="#673AB7")
    ax.set_title("ROAS Split By Age", fontsize=14)
    ax.set_xlabel("ROAS")
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height() / 2, f"{width:.2f}",
                va='center', fontsize=8)
    plt.tight_layout()
    return generate_chart_image(fig)

def generate_roas_split_by_gender_chart(df):
    # Standardize column names
    df = df.rename(columns={
        'Gender': 'gender',
        'ROAS': 'roas'
    })
    
    if 'gender' not in df.columns or 'roas' not in df.columns:
        raise ValueError("Required columns 'gender' and 'roas' not found")
    
    # Filter out invalid ROAS values
    df = df[df['roas'].notna() & (df['roas'] >= 0)]
    
    if df.empty:
        return create_empty_chart_image("No valid ROAS data by gender")
    
    grouped = df.groupby('gender')['roas'].mean()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(grouped.index, grouped.values, color="#673AB7")
    ax.set_title("ROAS Split By Gender", fontsize=14)
    ax.set_xlabel("ROAS")
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height() / 2, f"{width:.2f}",
                va='center', fontsize=8)
    plt.tight_layout()
    return generate_chart_image(fig)

def generate_platform_split_charts(df):
    df = df.copy()
    df['platform'] = df['platform'].fillna("Uncategorized")
    df['spend'] = pd.to_numeric(df['spend'], errors='coerce').fillna(0)
    df['purchase_value'] = pd.to_numeric(df['purchase_value'], errors='coerce').fillna(0)

    cost = df.groupby('platform')['spend'].sum().sort_values(ascending=False).head(5)
    revenue = df.groupby('platform')['purchase_value'].sum().sort_values(ascending=False).head(5)
    
    charts = []

    # Cost Split Pie
    # Add a check here for empty 'cost' series or all zeros
    if not cost.empty and cost.sum() > 0:
        fig1, ax1 = plt.subplots(figsize=(6, 6), dpi=200)
        ax1.pie(cost, labels=cost.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title("Cost Split by Platform")
        charts.append(("Cost Split", generate_chart_image(fig1)))
    else:
        # If no valid data, append a placeholder/empty chart image
        charts.append(("Cost Split", create_empty_chart_image("No Cost Data for Platforms")))


    # Revenue Split Pie
    # Add a check here for empty 'revenue' series or all zeros
    if not revenue.empty and revenue.sum() > 0:
        fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=200)
        ax2.pie(revenue, labels=revenue.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title("Revenue Split by Platform")
        charts.append(("Revenue Split", generate_chart_image(fig2)))
    else:
        # If no valid data, append a placeholder/empty chart image
        charts.append(("Revenue Split", create_empty_chart_image("No Revenue Data for Platforms")))

    return charts

def generate_platform_roas_chart(df):
    df = df.copy()
    df['platform'] = df['platform'].fillna("Uncategorized")
    df['spend'] = pd.to_numeric(df['spend'], errors='coerce').fillna(0)
    df['purchase_value'] = pd.to_numeric(df['purchase_value'], errors='coerce').fillna(0)

    platform_data = df.groupby('platform').agg({'spend': 'sum', 'purchase_value': 'sum'})
    
    # Filter out entries where spend is zero to avoid division by zero and NaN in ROAS
    platform_data = platform_data[platform_data['spend'] > 0]
    
    platform_data['roas'] = platform_data['purchase_value'] / platform_data['spend'].replace(0, 1)
    platform_data = platform_data.sort_values(by='spend', ascending=False).head(5)

    if platform_data.empty or platform_data['roas'].sum() <= 0:
        return create_empty_chart_image("No ROAS Data for Platforms")

    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
    ax.barh(platform_data.index, platform_data['roas'], color='#1f77b4')
    ax.set_title("ROAS by Platform")
    ax.set_xlabel("ROAS")
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    fig.tight_layout()

    return generate_chart_image(fig)

#def generate_platform_cost_line_chart(df):
    # df = df.copy()
    # df['date'] = pd.to_datetime(df['date'])
    # df['platform'] = df['platform'].fillna("Uncategorized")
    # df['spend'] = pd.to_numeric(df['spend'], errors='coerce').fillna(0)

    # grouped = df.groupby('platform')['spend'].sum().sort_values(ascending=False).head(5).index
    # df = df[df['platform'].isin(grouped)]

    # fig, ax = plt.subplots(figsize=(15, 6), dpi=200)
    # for column in df.columns:
    #     ax.plot(df.index, df[column], label=column, linewidth=2)

    # ax.set_title("Cost by Platform Over Time", fontsize=14)
    # ax.set_ylabel("Amount Spent")
    # ax.set_xlabel("Date")
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    # ax.tick_params(axis='x', rotation=45)
    # ax.grid(True, linestyle='--', alpha=0.3)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fontsize=8)

    # fig.tight_layout()
    # return ("Cost by Platform", generate_chart_image(fig))
    
def generate_platform_cost_line_chart(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date_start']) # Corrected to use 'date_start'
    df['platform'] = df['platform'].fillna("Uncategorized")
    df['spend'] = pd.to_numeric(df['spend'], errors='coerce').fillna(0)

    # Filter to top platforms by total spend
    top_platforms = df.groupby('platform')['spend'].sum().nlargest(5).index
    df_filtered = df[df['platform'].isin(top_platforms)]

    # Pivot to get spend per platform per day
    pivot_df = df_filtered.pivot_table(index='date', columns='platform', values='spend', aggfunc='sum').fillna(0)

    if pivot_df.empty or pivot_df.sum().sum() <= 0: # Check if pivot_df is empty or all zeros
        return create_empty_chart_image("No Cost Trend Data for Platforms")

    fig, ax = plt.subplots(figsize=(15, 6), dpi=200)
    for column in pivot_df.columns:
        ax.plot(pivot_df.index, pivot_df[column], label=column, linewidth=2)

    ax.set_title("Cost by Platform Over Time", fontsize=14)
    ax.set_ylabel("Amount Spent")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fontsize=8)

    fig.tight_layout()
    return generate_chart_image(fig)

def generate_platform_revenue_line_chart(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date_start']) # Corrected to use 'date_start'
    df['platform'] = df['platform'].fillna("Uncategorized")
    df['purchase_value'] = pd.to_numeric(df['purchase_value'], errors='coerce').fillna(0)

    # Filter to top platforms by total revenue
    top_platforms = df.groupby('platform')['purchase_value'].sum().nlargest(5).index
    df_filtered = df[df['platform'].isin(top_platforms)]

    # Pivot to get revenue per platform per day
    pivot_df = df_filtered.pivot_table(index='date', columns='platform', values='purchase_value', aggfunc='sum').fillna(0)

    if pivot_df.empty or pivot_df.sum().sum() <= 0: # Check if pivot_df is empty or all zeros
        return create_empty_chart_image("No Revenue Trend Data for Platforms")

    fig, ax = plt.subplots(figsize=(15, 6), dpi=200)
    for column in pivot_df.columns:
        ax.plot(pivot_df.index, pivot_df[column], label=column, linewidth=2)

    ax.set_title("Revenue by Platform Over Time", fontsize=14)
    ax.set_ylabel("Revenue")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fontsize=8)

    fig.tight_layout()
    return generate_chart_image(fig)

# def generate_platform_revenue_line_chart(df):
#     df = df.copy()
#     df['date'] = pd.to_datetime(df['date'])
#     df['platform'] = df['platform'].fillna("Uncategorized")
#     df['purchase_value'] = pd.to_numeric(df['purchase_value'], errors='coerce').fillna(0)

#     grouped = df.groupby(['platform', 'date'])['purchase_value'].sum().reset_index()
#     pivot_df = grouped.pivot(index='date', columns='platform', values='purchase_value').fillna(0)

#     fig, ax = plt.subplots(figsize=(15, 6), dpi=200)
#     for column in pivot_df.columns:
#         ax.plot(pivot_df.index, pivot_df[column], label=column, linewidth=2)

#     ax.set_title("Revenue by Platform Over Time", fontsize=14)
#     ax.set_ylabel("Revenue")
#     ax.set_xlabel("Date")
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
#     ax.tick_params(axis='x', rotation=45)
#     ax.grid(True, linestyle='--', alpha=0.3)
#     ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fontsize=8)

#     fig.tight_layout()
#     return ("Revenue by Platform", generate_chart_image(fig))
