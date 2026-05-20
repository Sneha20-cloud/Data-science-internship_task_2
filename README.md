# 📊 Task 2: Unemployment Analysis in India

An end-to-end data analysis project exploring unemployment trends in India, with a focus on the **Covid-19 impact**, regional disparities, and seasonal patterns.

---

## 📌 Objective
- Analyze unemployment rate data across Indian states (2019–2020)
- Investigate the impact of Covid-19 on unemployment rates
- Identify seasonal and regional patterns
- Present policy-relevant insights

---

## 📂 Project Structure
```
unemployment_analysis/
│
├── unemployment_analysis.py       # Main analysis script
├── plot1_national_trend.png       # National unemployment trend
├── plot2_region_unemployment.png  # Region-wise average unemployment
├── plot3_covid_impact.png         # Covid-19 impact comparison
├── plot4_heatmap.png              # Region × Month heatmap
├── plot5_seasonal.png             # Seasonal & year-on-year trends
├── plot6_regional_area.png        # Top/Bottom regions & Rural vs Urban
├── requirements.txt               # Python dependencies
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/unemployment-analysis.git
cd unemployment-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Dataset
- Download from [Kaggle](https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india)
- Place the CSV file in the **same folder** as `unemployment_analysis.py`

### 4. Run the script
```bash
python unemployment_analysis.py
```

> ⚠️ If no CSV is found, the script uses realistic demo data with the same structure.

---

## 📊 Dataset
- **Source**: [Kaggle — Unemployment in India](https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india)
- **Period**: Jan 2019 – Nov 2020
- **Features**: Region, Date, Unemployment Rate (%), Employed Count, Labour Participation Rate, Area (Rural/Urban)
- **Regions**: 28 Indian states

---

## 📈 Key Findings

| Period | Avg Unemployment Rate |
|--------|----------------------|
| Pre-Covid (2019 – Feb 2020) | ~7–8% |
| During Covid (Mar – Jul 2020) | ~23–26% 🔴 |
| Post-Covid Recovery (Aug – Nov 2020) | ~10–11% |

- **Covid-19 caused a 3× spike** in unemployment (Apr–May 2020 peak)
- **Rural areas** were disproportionately affected
- Significant **regional disparity** — some states recovered faster
- **Seasonal patterns** exist, tied to agricultural cycles

---

## 💡 Policy Recommendations
1. **Emergency Funds** — National buffer for rapid crisis response
2. **Rural Employment** — Expand MGNREGA-style schemes
3. **Skill Development** — Reskilling for displaced informal workers
4. **Regional Focus** — Targeted investment in high-unemployment states
5. **Seasonal Support** — Off-season alternatives for agri-dependent workers
6. **Data Monitoring** — Real-time monthly tracking for policy decisions

---

## 🛠️ Libraries Used
- `pandas` — Data loading, cleaning, manipulation
- `numpy` — Numerical operations
- `matplotlib` — All plots and charts
- `seaborn` — Heatmap visualization

---

## 📚 Concepts Learned
- Data cleaning (missing values, duplicates, type conversion)
- Exploratory Data Analysis (EDA)
- Time-series trend analysis
- Covid-19 pre/during/post period comparison
- Pivot tables and heatmaps
- Regional and seasonal pattern identification
- Translating data insights into policy recommendations

