# 💰 Gold Price Impact — LTV Analysis  
### Forecasting, Stress Testing & Risk Analytics Dashboard  
### *(A Fintech's-style FinTech Data Analyst Project)*  

---

## 📌 **Overview**

This project analyzes how **gold price movements impact the Loan-to-Value (LTV) ratio** for gold-backed loans — a core risk metric used by lenders like **Fintech's**, Muthoot, Manappuram, and banks operating under RBI's 75% LTV rule.

The project includes:

- 📈 45 years of gold price historical data  
- 🧹 Full data cleaning & transformation pipeline  
- 🔗 Merging multiple datasets  
- 🔮 90-day forecasting (Prophet / Holt-Winters)  
- 🧮 LTV simulation for 500 synthetic loans  
- ⚠️ Stress testing under -5%, -10%, -20% gold price shocks  
- 🌐 Streamlit dashboard for interactive risk analytics (Phase 6)  
- 📄 Detailed documentation (this README)

---

# 📁 Repository Name  
**`Gold-Price-Impact---LTV-Analysis`**

---

# 🌐 Live Dashboard (Optional)  
If deployed:  
**https://your-streamlit-app-url**

---

# 🧱 **Project Architecture Diagram**

    ┌─────────────────────────────────────────────────────┐
    │                RAW DATASETS                         │
    │ ─ Dataset 1: MCX Daily Gold Price (INR/10g)         │
    │ ─ Dataset 2: Multi-variant ML dataset (80 cols)     │
    │ ─ Dataset 3: WGC Historical Gold Prices (1978–2023) │
    └─────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌─────────────────────────────────────────────┐
        │       PHASE 2: CLEANING & TRANSFORMATION    │
        │ Convert Date, Handle NaNs, Ounce→10g, etc.  │
        └─────────────────────────────────────────────┘
                            │
                            ▼
        ┌─────────────────────────────────────────────┐
        │          PHASE 3: MERGING PIPELINE          │
        │ Combined historical + MCX recent data       │
        │ Built continuous INR/10g daily series       │
        └─────────────────────────────────────────────┘
                            │
                            ▼
        ┌─────────────────────────────────────────────┐
        │        PHASE 4: FORECAST ENGINE             │
        │ Prophet / Holt-Winters → 90-day forecast    │
        └─────────────────────────────────────────────┘
                            │
                            ▼
      ┌─────────────────────────────────────────────────┐
      │       PHASE 5: LTV CALCULATOR ENGINE            │
      │ Synthetic portfolio → LTV now + future + shocks │
      └─────────────────────────────────────────────────┘
                            │
                            ▼
      ┌────────────────────────────────────────────────┐
      │       PHASE 6: STREAMLIT DASHBOARD             │
      │ Interactive charts, filters, stress tests      │
      │ app.py → :contentReference[oaicite:1]{index=1} │
      └────────────────────────────────────────────────┘



---

# 🏗️ **Project Folder Structure**
```
Gold-Price-Impact---LTV-Analysis/
│
├── Dataset/
│ ├── dataset_2_1978–2023/
│ │ ├── Daily.csv
│ │ ├── Monthly_Avg.csv
│ │ ├── ...
│ ├── 1_Gold Price.csv
│ └── simulated_loan_portfolio.csv
│
├── gold_price_ts_daily.csv
├── gold_price_merged.csv
├── gold_price_forecast_90d.csv
├── gold_price_history_and_forecast.csv
│
├── gold_ltv_analysis.ipynb
├── app.py # Streamlit App (Dashboard)
└── README.md
```


---

# 🧠 **PHASE 1 — Dataset Analysis & Selection**

We began with **three datasets**:

### **Dataset 1 — MCX India Gold Price (2015–2025)**
| Pros | Cons |
|------|------|
| Accurate INR/10g data used by Indian lenders | Only 10 years of history |
| Matches Fintech's’s actual valuation model | Cannot model long-term seasonality |
| Clean, ready for LTV calculations | Not suitable alone for forecasting |

---

### **Dataset 2 — Multi-variant ML dataset (80 columns)**
| Pros | Cons |
|------|------|
| Contains Oil, S&P 500, USD Index, Silver, etc. | Overkill for pure data analytics |
| Best for ML regression models | No INR-specific price series |
| Great for future ML project | Hard to interpret for business stakeholders |

**→ Not used in this Data Analyst project (will be used for ML extension).**

---

### **Dataset 3 — WGC Historical Data (1978–2023)**
| Pros | Cons |
|------|------|
| 45 years of global gold prices | Some early INR values missing |
| Best for long-term trend & forecasting | Needs unit conversion from ounce |
| Official, authoritative | Requires transformation |

---

### 🎯 **Final Choice for Data Analyst Project: Dataset 1 + Dataset 3**

- **Dataset 1** = Accurate recent INR prices (LTV-ready)  
- **Dataset 3** = Long history for forecasting  

This combination provides:
- Realistic valuation model  
- Long-term forecasting ability  
- Clean INR per 10g continuity  
- High recruiter value  

---

# 🧹 **PHASE 2 — Data Cleaning & Processing**

### Key Steps:

### **2.1 Convert Date formats**
```python
df['Date'] = pd.to_datetime(df['Date'])
```

### **2.2 Convert INR per ounce → INR per gram → INR per 10g**
1 troy ounce = 31.1035 grams

INR_per_gram   = INR / 31.1035
INR_per_10g    = INR_per_gram * 10


### **2.3 Handle missing INR**
Forward fill:
```python
df['INR_per_10g'] = df['INR_per_10g'].fillna(method='ffill')
```

### **2.4 Sort & standardize**
```python
df = df.sort_values('Date')
```
---


# 🔗 **PHASE 3 — Merging Historical + MCX Prices**
Logic:
```
merged['Gold_Price'] = merged['Price_10g_mcx'].combine_first(merged['Price_10g_hist'])
```


Why?

- Use MCX if available

- Else fallback to historical data

- Ensures continuity from 1978 → 2025

Save final merged time series:
```
ts.to_csv("gold_price_ts_daily.csv", index=False)
```
---


# 🔮 **PHASE 4 — Forecast Engine**

We used:

- ✔ Prophet (if installed)
- ✔ Holt-Winters as fallback

### **Forecast Steps:**

 Prepare daily time series

- Fit forecasting model

- Generate 90-day forecast

- Save outputs:
    - gold_price_forecast_90d.csv

    - gold_price_history_and_forecast.csv

Example forecast visualization:
```
Historical ───────────────┐
                          ├── Forecast (yhat)
Lower/Upper Bands ────────┘
```
---


# 🧮 **PHASE 5 — LTV Engine & Stress Testing**
**Synthetic loan portfolio (500 loans)**

Saved as:
```
Dataset/simulated_loan_portfolio.csv
```
**LTV Formula**
```
gold_value_now = weight_g × purity_factor × current_price_per_gram

LTV_now = (loan_amount / gold_value_now) × 100
```

**Stress Scenarios:**

- Mild → -5%

- Medium → -10%

- Severe → -20%

**Output Summary:**

- LTV_now

- LTV_forecast

- LTV_-5%

- LTV_-10%

- LTV_-20%

- Risk flags (LTV > 75%)

---
# 🌐 **PHASE 6 — Streamlit Dashboard**


File: ```app.py```

📄app

## 🎨 **Dashboard Sections**

### 📈 **1. Gold Price Trend (History + Forecast)**
Plotly line chart:
```
px.line(combined, x="Date", y=["Historical","Forecast"])
```

Shows:

- Long-term trend

- Future forecast

- Confidence intervals

### 🏦 **2. Loan Portfolio Overview**
---
Metrics:

- Total loans

- Avg loan amount

- Avg LTV
### 📊 **3. LTV Distribution**
---
Histogram of current LTV:

- Red line at 75% RBI limit

- Helps spot overvalued loans
### ⚠️ **4. Stress Testing Section**
---
Sidebar option selects:

- None

- -5%

- -10%

- -20%

Updates:

- Recomputes LTV

- Shows # risky loans

- Visual distribution
### 🔥 **5. Top 20 Riskiest Loans**


Table shows:

- loan_id

- loan_amount

- gold_weight

- purity

- LTV_shock

----
##  📂 **Data Loading Logic in Streamlit**
(From app.py)
```
ts = pd.read_csv("gold_price_ts_daily.csv")
forecast = pd.read_csv("gold_price_forecast_90d.csv")
combined = pd.read_csv("gold_price_history_and_forecast.csv")
loan_df = pd.read_csv("Dataset/simulated_loan_portfolio.csv")
```
## **▶️ How to Run the Dashboard**
### **Install dependencies**
```
pip install streamlit pandas numpy plotly prophet statsmodels
```

### **Run Streamlit**
```
streamlit run app.py
```


Dashboard opens at:

👉 http://localhost:8501 ↗



# 📄 **Future Enhancements (Phase 7 / ML Extension)**

- Use Dataset 2 to build ML models for price prediction

- Train Random Forest, XGBoost, LSTM

- Add customer segmentation

- Add portfolio-level risk forecasting

- Deploy the dashboard on Streamlit Cloud

# ⭐ **Conclusion**

This project demonstrates end-to-end skills required for a FinTech Data Analyst / Risk Analyst role:

- Data engineering

- Time series forecasting

- Financial modeling

- LTV stress testing

- Interactive dashboards

- Documentation & storytelling

It aligns perfectly with companies like:    

- Fintech's

- Muthoot

- Manappuram

- Banks offering secured loans

# 🥷 **Author**

Name - *Ritesh Brahmachari*

- WebPage - https://riteshbrahmachariportfolio.vercel.app/

- LinkedIN - https://www.linkedin.com/in/ritesh-brahmachari-1b7b84278/

- Streamlit - https://share.streamlit.io/user/ritesh-456

---