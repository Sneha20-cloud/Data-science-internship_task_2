
# TASK 2: UNEMPLOYMENT ANALYSIS IN INDIA
# Internship Data Science Project
# Dataset: https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── Styling ──────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#f8f9fa',
    'axes.facecolor':   '#ffffff',
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'font.family':      'DejaVu Sans',
})
COLORS  = ['#2196F3','#FF5722','#4CAF50','#9C27B0','#FF9800',
           '#00BCD4','#E91E63','#607D8B','#8BC34A','#FFC107']
COVID_COLOR  = '#FF5722'
PRE_COLOR    = '#4CAF50'
DURING_COLOR = '#FF5722'
POST_COLOR   = '#2196F3'

print("=" * 65)
print("   TASK 2: UNEMPLOYMENT ANALYSIS IN INDIA")
print("=" * 65)

# ============================================================
# STEP 1 — Load Dataset
# ============================================================
print("\n[STEP 1] Loading Dataset...")

# ── Try to load the user's downloaded CSV (any common filename) ──
import os, glob

# Search for any CSV in the current directory or common locations
csv_candidates = (
    glob.glob("*.csv") +
    glob.glob("Unemployment*.csv") +
    glob.glob("unemployment*.csv") +
    glob.glob("**/*.csv", recursive=True)
)

if csv_candidates:
    filepath = csv_candidates[0]
    print(f"   Found CSV: {filepath}")
    df_raw = pd.read_csv(filepath)
else:
    # ── Fallback: realistic synthetic dataset matching Kaggle structure ──
    print("   CSV not found in current folder.")
    print("   ⚠  Using built-in demo data (same structure as Kaggle dataset).")
    print("   → To use real data: place your CSV in the same folder as this script.\n")

    np.random.seed(42)
    regions = [
        'Andhra Pradesh','Assam','Bihar','Delhi','Gujarat','Haryana',
        'Jharkhand','Karnataka','Kerala','Madhya Pradesh','Maharashtra',
        'Meghalaya','Odisha','Puducherry','Punjab','Rajasthan',
        'Sikkim','Tamil Nadu','Telangana','Tripura','Uttar Pradesh',
        'West Bengal'
    ]
    dates = pd.date_range(start='2019-01-31', end='2020-11-30', freq='MS')
    rows  = []
    for region in regions:
        base = np.random.uniform(3, 12)
        for d in dates:
            # Simulate Covid spike (Apr–Jul 2020)
            covid_bump = 0
            if d >= pd.Timestamp('2020-04-01') and d <= pd.Timestamp('2020-07-31'):
                covid_bump = np.random.uniform(15, 30)
            elif d >= pd.Timestamp('2020-08-01') and d <= pd.Timestamp('2020-11-30'):
                covid_bump = np.random.uniform(2, 8)
            seasonal = 2 * np.sin(2 * np.pi * d.month / 12)
            rate = base + seasonal + covid_bump + np.random.normal(0, 0.8)
            rate = max(0, rate)
            area  = 'Rural' if np.random.random() > 0.5 else 'Urban'
            labour = np.random.randint(10_000_000, 50_000_000)
            employed = int(labour * (1 - rate/100))
            rows.append({
                'Region':                       region,
                'Date':                         d,
                'Frequency':                    'Monthly',
                'Estimated Unemployment Rate (%)': round(rate, 2),
                'Estimated Employed':           employed,
                'Estimated Labour Participation Rate (%)': round(np.random.uniform(35, 55), 2),
                'Area':                         area,
            })
    df_raw = pd.DataFrame(rows)

print(f"   Shape: {df_raw.shape}")
print(f"\n   Columns: {list(df_raw.columns)}")
print(f"\n   First 5 rows:\n{df_raw.head()}")

# ============================================================
# STEP 2 — Data Cleaning
# ============================================================
print("\n[STEP 2] Cleaning Data...")

df = df_raw.copy()

# Standardise column names
df.columns = df.columns.str.strip()

# Identify key columns flexibly
col_map = {}
for col in df.columns:
    cl = col.lower()
    if 'region' in cl or 'state' in cl:         col_map['region']     = col
    elif 'date' in cl:                           col_map['date']       = col
    elif 'unemployment' in cl and '%' in cl:     col_map['unemp_rate'] = col
    elif 'unemployment' in cl and 'rate' in cl:  col_map['unemp_rate'] = col
    elif 'employed' in cl and 'estimated' in cl \
         and 'un' not in cl:                     col_map['employed']   = col
    elif 'labour' in cl or 'labor' in cl:        col_map['labour']     = col
    elif 'area' in cl:                           col_map['area']       = col

print(f"   Column mapping detected: {col_map}")

# Parse dates
df[col_map['date']] = pd.to_datetime(df[col_map['date']], dayfirst=True, errors='coerce')
df = df.dropna(subset=[col_map['date']])
df = df.sort_values(col_map['date']).reset_index(drop=True)

# Drop duplicates & check nulls
before = len(df)
df = df.drop_duplicates()
print(f"   Duplicates removed: {before - len(df)}")
print(f"   Missing values:\n{df.isnull().sum()}")
df = df.dropna(subset=[col_map['unemp_rate']])

# Add helper columns
df['Year']  = df[col_map['date']].dt.year
df['Month'] = df[col_map['date']].dt.month
df['Month_Name'] = df[col_map['date']].dt.strftime('%b')

# Tag Covid periods
def covid_period(d):
    if d < pd.Timestamp('2020-03-01'):  return 'Pre-Covid'
    if d <= pd.Timestamp('2020-07-31'): return 'During Covid'
    return 'Post-Covid (Recovery)'

df['Period'] = df[col_map['date']].apply(covid_period)

print(f"\n   Date range : {df[col_map['date']].min().date()} → {df[col_map['date']].max().date()}")
print(f"   Regions    : {df[col_map['region']].nunique()}")
print(f"   Records    : {len(df)}")

# Short aliases for readability
R  = col_map['region']
D  = col_map['date']
UR = col_map['unemp_rate']
LB = col_map.get('labour', None)

# ============================================================
# STEP 3 — Exploratory Data Analysis
# ============================================================
print("\n[STEP 3] Exploratory Data Analysis...")

print(f"\n   Overall Unemployment Stats:")
print(df[UR].describe().round(2))

print(f"\n   Average Unemployment by Period:")
print(df.groupby('Period')[UR].mean().round(2))

print(f"\n   Top 5 Regions (Highest Avg Unemployment):")
top5 = df.groupby(R)[UR].mean().sort_values(ascending=False).head(5)
print(top5.round(2))

# ============================================================
# STEP 4 — Visualisation 1: National Trend + Covid Overlay
# ============================================================
print("\n[STEP 4] Plot 1 — National Unemployment Trend...")

monthly_avg = df.groupby(D)[UR].mean().reset_index()

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('#f8f9fa')

ax.fill_between(monthly_avg[D], monthly_avg[UR], alpha=0.15, color='#2196F3')
ax.plot(monthly_avg[D], monthly_avg[UR], color='#2196F3', lw=2.5, marker='o', ms=5, label='Unemployment Rate')

# Covid shading
covid_start = pd.Timestamp('2020-03-01')
covid_end   = pd.Timestamp('2020-07-31')
ax.axvspan(covid_start, covid_end, alpha=0.15, color='red', label='Covid-19 Peak Period')
ax.axvline(covid_start, color='red', lw=1.5, ls='--', alpha=0.7)

# Annotate peak
peak_idx = monthly_avg[UR].idxmax()
peak_val = monthly_avg[UR].max()
peak_date= monthly_avg.loc[peak_idx, D]
ax.annotate(f'Peak: {peak_val:.1f}%',
            xy=(peak_date, peak_val),
            xytext=(peak_date - pd.DateOffset(months=2), peak_val + 2),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=11, color='red', fontweight='bold')

ax.set_title('India — National Unemployment Rate Trend\n(with Covid-19 Impact)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Unemployment Rate (%)', fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('plot1_national_trend.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Saved: plot1_national_trend.png")

# ============================================================
# STEP 5 — Visualisation 2: Region-wise Avg Unemployment
# ============================================================
print("\n[STEP 5] Plot 2 — Region-wise Unemployment...")

region_avg = df.groupby(R)[UR].mean().sort_values(ascending=False).reset_index()

fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.barh(region_avg[R], region_avg[UR],
               color=[COLORS[i % len(COLORS)] for i in range(len(region_avg))],
               edgecolor='white', height=0.7)
for bar, val in zip(bars, region_avg[UR]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')

ax.set_title('Average Unemployment Rate by Region', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Average Unemployment Rate (%)', fontsize=11)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('plot2_region_unemployment.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Saved: plot2_region_unemployment.png")

# ============================================================
# STEP 6 — Visualisation 3: Covid-19 Impact Comparison
# ============================================================
print("\n[STEP 6] Plot 3 — Covid-19 Impact Analysis...")

period_stats = df.groupby('Period')[UR].agg(['mean','median','max','min']).round(2)
period_order = ['Pre-Covid','During Covid','Post-Covid (Recovery)']
period_stats = period_stats.reindex([p for p in period_order if p in period_stats.index])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Covid-19 Impact on Unemployment Rates', fontsize=14, fontweight='bold')

# Bar chart — mean by period
period_colors = [PRE_COLOR, DURING_COLOR, POST_COLOR]
bars = axes[0].bar(period_stats.index, period_stats['mean'],
                   color=period_colors[:len(period_stats)],
                   edgecolor='black', alpha=0.85)
for bar, val in zip(bars, period_stats['mean']):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)
axes[0].set_title('Average Unemployment by Period', fontsize=12)
axes[0].set_ylabel('Avg Unemployment Rate (%)')
axes[0].set_xticklabels(period_stats.index, rotation=10, ha='right')

# Box plot — distribution by period
period_data = [df[df['Period'] == p][UR].values
               for p in period_order if p in df['Period'].unique()]
period_labels = [p for p in period_order if p in df['Period'].unique()]
bp = axes[1].boxplot(period_data, labels=period_labels, patch_artist=True,
                     medianprops=dict(color='black', lw=2))
for patch, color in zip(bp['boxes'], period_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_title('Unemployment Distribution by Period', fontsize=12)
axes[1].set_ylabel('Unemployment Rate (%)')
axes[1].set_xticklabels(period_labels, rotation=10, ha='right')

plt.tight_layout()
plt.savefig('plot3_covid_impact.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Saved: plot3_covid_impact.png")

# ============================================================
# STEP 7 — Visualisation 4: Heatmap — Region × Month
# ============================================================
print("\n[STEP 7] Plot 4 — Heatmap (Region vs Month)...")

pivot = df.pivot_table(values=UR, index=R, columns='Month_Name', aggfunc='mean')
month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])

fig, ax = plt.subplots(figsize=(14, 9))
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', linewidths=0.5,
            linecolor='white', ax=ax, cbar_kws={'label': 'Unemployment Rate (%)'})
ax.set_title('Unemployment Rate Heatmap: Region × Month', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Month', fontsize=11)
ax.set_ylabel('Region', fontsize=11)
plt.tight_layout()
plt.savefig('plot4_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Saved: plot4_heatmap.png")

# ============================================================
# STEP 8 — Visualisation 5: Seasonal / Monthly Trend
# ============================================================
print("\n[STEP 8] Plot 5 — Seasonal Monthly Trend...")

monthly_trend = df.groupby('Month')[UR].mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Seasonal & Monthly Unemployment Patterns', fontsize=14, fontweight='bold')

# Line plot
month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
axes[0].plot(range(1, len(monthly_trend)+1), monthly_trend.values,
             marker='o', color='#2196F3', lw=2.5, ms=8)
axes[0].fill_between(range(1, len(monthly_trend)+1), monthly_trend.values, alpha=0.1, color='#2196F3')
axes[0].set_xticks(range(1, len(monthly_trend)+1))
axes[0].set_xticklabels(month_labels[:len(monthly_trend)], rotation=45)
axes[0].set_title('Average Unemployment by Month', fontsize=12)
axes[0].set_ylabel('Avg Unemployment Rate (%)')

# Year-on-year comparison
yearly = df.groupby('Year')[UR].mean()
bar_colors = ['#4CAF50' if y < 2020 else '#FF5722' for y in yearly.index]
bars = axes[1].bar(yearly.index.astype(str), yearly.values,
                   color=bar_colors, edgecolor='black', alpha=0.85)
for bar, val in zip(bars, yearly.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.1f}%', ha='center', fontweight='bold')
axes[1].set_title('Year-on-Year Comparison', fontsize=12)
axes[1].set_ylabel('Avg Unemployment Rate (%)')
green_patch = mpatches.Patch(color='#4CAF50', label='Pre-Covid')
red_patch   = mpatches.Patch(color='#FF5722', label='Covid Year')
axes[1].legend(handles=[green_patch, red_patch])

plt.tight_layout()
plt.savefig('plot5_seasonal.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Saved: plot5_seasonal.png")

# ============================================================
# STEP 9 — Visualisation 6: Top vs Bottom Regions + Area
# ============================================================
print("\n[STEP 9] Plot 6 — Top/Bottom Regions & Area Comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Regional Extremes & Urban vs Rural Unemployment', fontsize=14, fontweight='bold')

# Top 5 vs Bottom 5 regions
region_avg2 = df.groupby(R)[UR].mean().sort_values(ascending=False)
top5_r  = region_avg2.head(5)
bot5_r  = region_avg2.tail(5)

x  = np.arange(5)
w  = 0.35
axes[0].bar(x - w/2, top5_r.values,  width=w, label='Highest 5', color='#FF5722', alpha=0.85, edgecolor='black')
axes[0].bar(x + w/2, bot5_r.values,  width=w, label='Lowest 5',  color='#4CAF50', alpha=0.85, edgecolor='black')
axes[0].set_xticks(x)
axes[0].set_xticklabels([f'R{i+1}' for i in range(5)])
axes[0].legend()
axes[0].set_title('Top 5 vs Bottom 5 Regions', fontsize=12)
axes[0].set_ylabel('Avg Unemployment Rate (%)')

top_names = '\n'.join([f'R{i+1}: {n}' for i, n in enumerate(top5_r.index)])
bot_names = '\n'.join([f'R{i+1}: {n}' for i, n in enumerate(bot5_r.index)])
axes[0].text(0.01, 0.99, f"Highest:\n{top_names}", transform=axes[0].transAxes,
             fontsize=7, va='top', color='#FF5722')

# Rural vs Urban (if area column exists)
if 'area' in col_map:
    A = col_map['area']
    area_avg = df.groupby(A)[UR].mean()
    bars = axes[1].bar(area_avg.index, area_avg.values,
                       color=['#9C27B0','#FF9800'], edgecolor='black', alpha=0.85)
    for bar, val in zip(bars, area_avg.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f'{val:.1f}%', ha='center', fontweight='bold', fontsize=12)
    axes[1].set_title('Rural vs Urban Unemployment', fontsize=12)
    axes[1].set_ylabel('Avg Unemployment Rate (%)')
else:
    axes[1].text(0.5, 0.5, 'Area column not found\nin dataset', ha='center', va='center',
                 transform=axes[1].transAxes, fontsize=12)

plt.tight_layout()
plt.savefig('plot6_regional_area.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("   Saved: plot6_regional_area.png")

# ============================================================
# STEP 10 — Policy Insights Summary
# ============================================================
print("\n" + "=" * 65)
print("   STEP 10 — KEY INSIGHTS & POLICY RECOMMENDATIONS")
print("=" * 65)

pre_avg    = df[df['Period'] == 'Pre-Covid'][UR].mean()
during_avg = df[df['Period'] == 'During Covid'][UR].mean()
post_avg   = df[df['Period'] == 'Post-Covid (Recovery)'][UR].mean()
peak       = df[UR].max()
worst      = df.groupby(R)[UR].mean().idxmax()
best       = df.groupby(R)[UR].mean().idxmin()

print(f"""
📊 FINDINGS:
   • Pre-Covid avg unemployment  : {pre_avg:.2f}%
   • During Covid avg            : {during_avg:.2f}%  (+{during_avg - pre_avg:.2f}% spike)
   • Post-Covid avg (recovery)   : {post_avg:.2f}%
   • Peak unemployment recorded  : {peak:.2f}%
   • Worst affected region       : {worst}
   • Best performing region      : {best}

💡 POLICY RECOMMENDATIONS:
   1. EMERGENCY FUNDS   — Maintain a national unemployment buffer
                          fund to deploy rapidly during crises.
   2. RURAL EMPLOYMENT  — Expand MGNREGA-like schemes; rural areas
                          showed higher vulnerability.
   3. SKILL DEVELOPMENT — Post-Covid recovery needs reskilling
                          programs for displaced informal workers.
   4. REGIONAL FOCUS    — States like {worst} need targeted
                          industrial investment & job creation.
   5. SEASONAL SUPPORT  — Agri-dependent states need off-season
                          employment alternatives.
   6. DATA MONITORING   — Monthly unemployment tracking should
                          inform real-time policy decisions.
""")

print("=" * 65)
print("   ALL PLOTS SAVED — Task 2 Complete! ✅")
print("=" * 65)
print("""
Files generated:
  • plot1_national_trend.png
  • plot2_region_unemployment.png
  • plot3_covid_impact.png
  • plot4_heatmap.png
  • plot5_seasonal.png
  • plot6_regional_area.png
""")
