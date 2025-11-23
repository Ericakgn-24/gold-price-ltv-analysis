import os
import json
from datetime import timedelta
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# optional imports (GPU & ML libs)
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import shap
except Exception:
    shap = None

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

try:
    from streamlit_lottie import st_lottie
    import requests
except ImportError:
    st_lottie = None

def load_lottieurl(url: str):
    if st_lottie is None:
        return None
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# -----------------------
# Paths / constants
# -----------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Dataset/processed_dataset_2_ml.csv"
PORTFOLIO_PATH = BASE_DIR / "Dataset/simulated_loan_portfolio.csv"
RF_MODEL_PATH = BASE_DIR / "models/random_forest_model.pkl"
XGB_MODEL_PATH = BASE_DIR / "models/xgb_model.json"   # prefer JSON for XGBoost v3
LSTM_MODEL_PATH = BASE_DIR / "models/lstm_model.pt"
RF_FEATURES_PATH = BASE_DIR / "models/rf_feature_names.json"

# ensure directories exist
(BASE_DIR / "models").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "outputs").mkdir(parents=True, exist_ok=True)

st.set_page_config(layout="wide", page_title="Gold Price LTV — ML Dashboard")

# -----------------------
# Custom CSS & Animations
# -----------------------
st.markdown("""
<style>
    /* Modern Dark Theme Gradient */
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2b42 50%, #2b1c3d 100%);
        color: #ffffff;
    }
    
    /* Glassmorphism Card Style */
    div[data-testid="stMetric"], div[data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 15px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
        border-color: rgba(255, 215, 0, 0.5); /* Gold border on hover */
    }
    
    /* Advanced Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, #ff8c00, #ff0080);
        border: none;
        color: white;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 30px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(255, 0, 128, 0.6);
        background: linear-gradient(90deg, #ff0080, #ff8c00); /* Reverse gradient */
    }
    .stButton > button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(255,255,255,0.05);
        padding: 10px;
        border-radius: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 15px;
        color: #ffffff;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.15) !important;
        color: #ffd700 !important; /* Gold text for active tab */
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    
    /* Headers & Text */
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #ffd700, #ff8c00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }
    p, label, span, div {
        color: #e0e0e0 !important;
    }
    
    /* Sidebar Polish */
    section[data-testid="stSidebar"] {
        background-color: rgba(30, 30, 47, 0.95);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load Animations
lottie_gold = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_kuhijlvx.json") # Gold/Finance
lottie_rocket = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_9wjm14ni.json") # Rocket/Growth

# Sidebar Animation
with st.sidebar:
    if lottie_gold:
        st_lottie(lottie_gold, height=150, key="sidebar_anim")
    else:
        st.image("https://cdn-icons-png.flaticon.com/512/2534/2534204.png", width=100)

# Main Title with Animation
col_title, col_anim = st.columns([3, 1])
with col_title:
    st.title("Gold Price Impact & LTV — ML Dashboard")
    st.markdown("Multi-tab dashboard: **LTV Analysis**, **ML Forecast & Predict**, **Explainability (SHAP)**")
with col_anim:
    if lottie_rocket:
        st_lottie(lottie_rocket, height=120, key="header_anim")

# -----------------------
# Helpers: load dataset & models
# -----------------------
@st.cache_data(ttl=3600)
def load_processed_data(path: Path = DATA_PATH):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        return df
    except Exception as e:
        st.error(f"Error reading processed dataset: {e}")
        return None

@st.cache_resource
def load_rf_model(path: Path = RF_MODEL_PATH):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load RF model: {e}")
        return None

@st.cache_resource
def load_xgb_booster(path: Path = XGB_MODEL_PATH):
    if xgb is None:
        return None
    if not path.exists():
        return None
    try:
        booster = xgb.Booster()
        booster.load_model(str(path))
        return booster
    except Exception as e:
        st.warning(f"XGBoost load failed: {e}")
        return None

@st.cache_resource
def load_lstm(path: Path = LSTM_MODEL_PATH):
    if torch is None:
        return None
    if not path.exists():
        return None
    try:
        # user must implement LSTM class in lstm_model_def.py or similar
        from lstm_model_def import LSTMModel  # user provided
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMModel(input_dim=97)  # matches checkpoint features
        model.load_state_dict(torch.load(str(path), map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.warning(f"LSTM load failed: {e}")
        return None

def load_rf_feature_names(path: Path = RF_FEATURES_PATH, df: pd.DataFrame | None = None):
    """
    Return the exact feature list used during RF training.
    If the JSON is missing or invalid, try to infer a safe fallback from the
    processed dataset (df) by selecting numeric columns excluding target.
    """
    if path.exists():
        try:
            with open(path, "r") as f:
                features = json.load(f)
                if isinstance(features, list) and len(features) > 0:
                    return features
        except Exception:
            st.warning("rf_feature_names.json found but invalid — regenerating from dataset (if possible).")

    # fallback: if df provided, attempt to infer (drop target "Adj Close")
    if df is not None:
        cols = [c for c in df.columns if c != "Adj Close" and pd.api.types.is_numeric_dtype(df[c])]
        # Keep order consistent
        return cols
    return None

# -----------------------
# Utility: build X_input matching model features
# -----------------------
def build_input_from_last_row(df: pd.DataFrame, rf_features: list, edits: dict):
    """
    Given the processed dataset (df), build a single-row DataFrame that contains
    the exact features rf_features in the same order the model expects.
    edits: mapping of feature->value from the UI override.
    """
    if df is None or rf_features is None:
        return None
    last_row = df.iloc[-1].copy()
    # We'll build X_input dict only from rf_features (ignore others)
    X_input = {}
    for f in rf_features:
        if f in last_row:
            X_input[f] = last_row[f]
        else:
            # If missing in last_row, fill with NaN
            X_input[f] = np.nan

    # Apply overrides from edits (only when feature is in rf_features)
    for k, v in (edits or {}).items():
        if k in X_input:
            try:
                X_input[k] = float(v)
            except Exception:
                # keep original if conversion fails
                pass

    # return DataFrame with single row and columns in the exact order
    return pd.DataFrame([X_input], columns=rf_features)

# -----------------------
# UI
# -----------------------
# st.title("Gold Price Impact & LTV — ML Dashboard")  <-- Replaced by animated header above
# st.markdown("Multi-tab dashboard: **LTV Analysis**, **ML Forecast & Predict**, **Explainability (SHAP)**")

# load processed dataset
df = load_processed_data()
if df is None:
    st.warning(f"Processed dataset not found at `{DATA_PATH}`. Run Phase 8 to produce it.")
else:
    st.success(f"Processed ML Dataset loaded: {df.shape}")

# load models & features
rf_model = load_rf_model()
xgb_booster = load_xgb_booster()
lstm_model = load_lstm()
rf_features = load_rf_feature_names(RF_FEATURES_PATH, df)

# show top-of-app model summary
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("RF Model", "Loaded" if rf_model is not None else "Missing")
with col2:
    st.metric("XGBoost Booster", "Loaded" if xgb_booster is not None else "Missing")
with col3:
    st.metric("LSTM Model", "Loaded" if lstm_model is not None else "Missing")

# -----------------------
# Tabbed layout
# -----------------------
tab = st.tabs(["LTV Analysis", "ML Forecast & Predict", "Explainability (SHAP)"])
tab0, tab1, tab2 = tab

# -----------------------
# Tab 0: LTV Analysis (simple table if portfolio exists)
# -----------------------
with tab0:
    st.header("LTV Analysis & Portfolio (Data Analyst)")
    if not Path(PORTFOLIO_PATH).exists():
        st.info("Simulated portfolio not found. Create it in Phase 6 or place it at Dataset/simulated_loan_portfolio.csv.")
    else:
        try:
            port = pd.read_csv(PORTFOLIO_PATH, parse_dates=["disbursal_date"])
            st.dataframe(port.head(50))
        except Exception as e:
            st.error(f"Failed to load portfolio: {e}")

# -----------------------
# Tab 1: ML Forecast & Predict
# -----------------------
with tab1:
    st.header("ML Forecast & Single-day Predictions")
    if df is None:
        st.info("Processed dataset missing.")
    else:
        # quick dataset preview
        st.subheader("Processed dataset (preview)")
        st.dataframe(df.head(6))

        st.markdown("---")
        st.subheader("Single-day prediction (use last row features or edit)")

        # Collect small set of editable numeric fields (subset of rf_features)
        edits = {}
        if rf_features is None:
            st.warning("RF feature list not found. Can't build prediction input.")
        else:
            # show a few editable controls for top features (first 8)
            preview_features = rf_features[:8]
            with st.form("edit_form"):
                cols = st.columns(len(preview_features))
                for i, f in enumerate(preview_features):
                    val = float(df.iloc[-1].get(f, np.nan)) if f in df.columns else 0.0
                    with cols[i]:
                        edits[f] = st.number_input(f, value=float(val), format="%.6f")
                submitted = st.form_submit_button("Apply edits")
                if submitted:
                    st.success("Overrides applied.")

            # Build final input matching RF feature list exactly
            X_input_df = build_input_from_last_row(df, rf_features, edits)
            st.write("Input preview (first 10 columns):")
            st.dataframe(X_input_df.iloc[:, :10].T)

            # Allow model selection
            model_choice = st.selectbox("Choose model for prediction", ["Random Forest", "XGBoost", "LSTM"])
            if st.button("Predict single day"):
                if model_choice == "Random Forest":
                    if rf_model is None:
                        st.error("Random Forest model not found at models/random_forest_model.pkl")
                    else:
                        try:
                            pred = rf_model.predict(X_input_df)[0]
                            st.metric("RF Predicted Adj Close", f"{float(pred):.2f}")
                        except Exception as e:
                            st.error(f"RF prediction failed: {e}")

                elif model_choice == "XGBoost":
                    if xgb_booster is None:
                        st.error("XGBoost booster not loaded (check models/xgb_model.json and xgboost installation).")
                    else:
                        try:
                            # use DMatrix with correct feature order
                            dX = xgb.DMatrix(X_input_df.values, feature_names=X_input_df.columns.tolist())
                            pred = xgb_booster.predict(dX)
                            st.metric("XGB Predicted Adj Close", f"{float(pred[0]):.2f}")
                        except Exception as e:
                            st.error(f"XGB prediction failed: {e}")

                else:  # LSTM
                    if lstm_model is None or torch is None:
                        st.error("LSTM model not available or torch not installed.")
                    else:
                        try:
                            # LSTM trained on 97 features (all numeric except target), unlike RF/XGB (79 features)
                            seq_len = 30
                            # Always use all available features for LSTM as per inspection
                            feat_cols = [c for c in df.columns if c != "Adj Close"]
                            
                            # Ensure we have enough data
                            if len(df) < seq_len:
                                st.error(f"Not enough data for LSTM (needs {seq_len} days).")
                            else:
                                seq_vals = df[feat_cols].values[-seq_len:]
                                # Explicitly convert to float32 numpy array first
                                seq_vals = seq_vals.astype(np.float32)
                                
                                # convert to tensor (1, seq_len, n_features)
                                x = torch.tensor(seq_vals).unsqueeze(0)  # (1, seq_len, n_features)
                                device = next(lstm_model.parameters()).device
                                x = x.to(device)
                                with torch.no_grad():
                                    out = lstm_model(x)   # assume model returns batch of scalars
                                    val = out.cpu().numpy().squeeze().item()
                                st.metric("LSTM Predicted Adj Close", f"{val:.2f}")
                        except Exception as e:
                            st.error(f"LSTM prediction failed: {e}")

        st.markdown("---")
        st.subheader("90-day Forecast (simple CPU fallback)")
        # simple CPU prognostic using last known Adj Close trend (naive)
        if st.button("Run quick naive forecast"):
            try:
                last = df["Adj Close"].iloc[-1]
                # naive: simple drift equal to last daily change mean (30d)
                drift = df["Adj Close"].diff().rolling(30).mean().iloc[-1]
                days = st.slider("Forecast days", min_value=7, max_value=180, value=90)
                future_idx = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days)
                preds = [last + (i + 1) * drift for i in range(days)]
                forecast_df = pd.DataFrame({"Date": future_idx, "Adj_Close_forecast": preds}).set_index("Date")
                st.line_chart(forecast_df)
            except Exception as e:
                st.error(f"Forecast failed: {e}")

# -----------------------
# Tab 2: Explainability (SHAP)
# -----------------------
with tab2:
    st.header("Explainability & Feature Importance")
    if df is None:
        st.info("Processed dataset missing.")
    elif xgb_booster is None and rf_model is None:
        st.info("No model available for SHAP explanations.")
    elif shap is None:
        st.warning("SHAP not installed — explanations not available. Install shap to enable this feature.")
    else:
        st.markdown("Compute SHAP values for a sample (this may take time).")
        # prefer RF if available (tree explainer works)
        # prefer RF if available (tree explainer works)
        try:
            with st.spinner("Computing SHAP values... (this may take a moment)"):
                sample = df.sample(min(500, len(df)), random_state=42)
                # Filter features to match the model's expectation (79 features for RF/XGB)
                if rf_features:
                    shap_cols = [c for c in rf_features if c in df.columns]
                else:
                    shap_cols = [c for c in df.columns if c != "Adj Close"]
                
                X_shap = sample[shap_cols]
                
                if rf_model is not None:
                    explainer = shap.TreeExplainer(rf_model)
                    shap_vals = explainer.shap_values(X_shap)
                    
                    # Handle case where shap_vals is a list (e.g. some versions/models)
                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[0]
                        
                    mean_abs = np.mean(np.abs(shap_vals), axis=0)
                    feat_imp = pd.DataFrame({"feature": X_shap.columns, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
                    st.dataframe(feat_imp.head(25))
                    
                    fig, ax = plt.subplots()
                    shap.summary_plot(shap_vals, X_shap, show=False)
                    st.pyplot(fig)
                    
                elif xgb_booster is not None:
                    explainer = shap.TreeExplainer(xgb_booster)
                    shap_vals = explainer.shap_values(X_shap)
                    
                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[0]
                        
                    mean_abs = np.mean(np.abs(shap_vals), axis=0)
                    feat_imp = pd.DataFrame({"feature": X_shap.columns, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
                    st.dataframe(feat_imp.head(25))
                    
                    fig, ax = plt.subplots()
                    shap.summary_plot(shap_vals, X_shap, show=False)
                    st.pyplot(fig)
                    
        except Exception as e:
            st.error(f"SHAP explanation failed: {e}")
            # st.exception(e) # Optional: show traceback

# -----------------------
# Save RF feature list helper (if user wants to generate)
# -----------------------
st.sidebar.header("Tools")
if st.sidebar.button("Regenerate rf_feature_names.json from dataset"):
    if df is None:
        st.sidebar.error("Processed dataset missing.")
    else:
        inferred = [c for c in df.columns if c != "Adj Close" and pd.api.types.is_numeric_dtype(df[c])]
        with open(RF_FEATURES_PATH, "w") as f:
            json.dump(inferred, f)
        st.sidebar.success(f"rf_feature_names.json saved with {len(inferred)} features.")

# st.sidebar.markdown("**Model paths**")
# st.sidebar.write(f"RF: {RF_MODEL_PATH}")
# st.sidebar.write(f"XGB: {XGB_MODEL_PATH}")
# st.sidebar.write(f"LSTM: {LSTM_MODEL_PATH}")

# small footer
st.markdown("---")
st.caption("Notes: 1) RF feature names must exactly match what the model was trained on. 2) For XGBoost use JSON/UBJ models for XGBoost 3.x+. 3) For LSTM ensure lstm_model_def.py exists and LSTMModel class matches saved weights.")
