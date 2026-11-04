from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODELS = ["xgboost", "lightgbm", "random_forest", "neural_baseline"]
MODEL_LABELS = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "random_forest": "Random Forest", "neural_baseline": "Neural Baseline"}
TICKERS = {
    "Reliance Industries": "RELIANCE.NS", "Tata Consultancy Services": "TCS.NS",
    "Infosys": "INFY.NS", "HDFC Bank": "HDFCBANK.NS", "State Bank of India": "SBIN.NS"
}

st.set_page_config(page_title="StockGPT | Explainable Market Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --bg:#080d18; --panel:#101827; --line:#202d40; --muted:#8b9ab0; --text:#e8eef8; --green:#4adea3; --cyan:#5ed8ed; }
html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
.stApp { background: radial-gradient(ellipse at 75% -15%, #152b3d 0%, #080d18 48%); color:var(--text); }
[data-testid="stSidebar"] { background:#0b1321; border-right:1px solid #1d2a3d; }
.block-container { padding-top:2rem; max-width:1500px; }
.eyebrow { color:#6fdfc0; font:500 11px 'DM Mono'; letter-spacing:2px; text-transform:uppercase; }
.hero { padding:8px 0 22px 0; border-bottom:1px solid #202d40; margin-bottom:24px; }
.hero h1 { font-size:34px; line-height:1.15; letter-spacing:-1.4px; margin:8px 0 6px; color:#f2f6fc; }
.hero p { color:#93a3b8; margin:0; font-size:14px; }
.metric { background:linear-gradient(145deg,#121f30,#0e1725); border:1px solid #26354a; border-radius:13px; padding:17px 18px; min-height:112px; }
.metric .label { color:#8fa0b6; font-size:11px; text-transform:uppercase; letter-spacing:1.1px; }
.metric .value { color:#f0f5fb; font-size:27px; font-weight:800; margin-top:8px; letter-spacing:-.8px; }
.metric .sub { color:#8294aa; font-size:11px; margin-top:3px; }
.section-title { font-size:17px; font-weight:800; color:#edf3fb; margin:12px 0 4px; }
.section-sub { color:#8494aa; font-size:12px; margin-bottom:14px; }
.note { border:1px solid #2a3a50; background:#111d2c; border-radius:10px; padding:12px 14px; color:#aebdd0; font-size:12px; }
.signal { display:inline-block; border:1px solid #28634f; background:#102b27; color:#6ee7b7; border-radius:99px; padding:5px 10px; font:11px 'DM Mono'; }
div[data-testid="stTabs"] button { color:#9eacc0; }
.stButton button { border-radius:9px; border:1px solid #2d8069; background:#12382f; color:#a5f3d1; font-weight:700; }
.stButton button:hover { border-color:#5ee5b6; color:white; }
[data-testid="stDataFrame"] { border:1px solid #26354a; border-radius:10px; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


def load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def metric_card(label, value, sub=""):
    st.markdown(f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def run_script(args, label):
    with st.spinner(f"Running {label}... this may take a while"):
        proc = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    if proc.returncode == 0:
        st.success(f"{label} completed.")
        if proc.stdout:
            st.code(proc.stdout[-7000:])
    else:
        st.error(f"{label} failed (exit code {proc.returncode}).")
        st.code((proc.stderr or proc.stdout)[-7000:])
    return proc.returncode == 0


def get_metrics(model):
    return load_json(ROOT / "models" / "prediction" / model / "metrics.json", {}) or {}


def get_xai(model):
    return load_json(ROOT / "data" / "explanations" / model / "global_feature_importance.json", []) or []


def chart_layout(fig, height=330):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#aebbd0", margin=dict(l=8,r=8,t=25,b=8),
                      xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(gridcolor="#243247", zeroline=False))
    return fig

with st.sidebar:
    st.markdown("<div class='eyebrow'>◈ STOCKGPT / RESEARCH TERMINAL</div>", unsafe_allow_html=True)
    st.markdown("## Navigation")
    page = st.radio("", ["Overview", "Market Explorer", "Explainability", "Model Lab", "Pipeline Control"], label_visibility="collapsed")
    st.divider()
    st.markdown("<div class='eyebrow'>WORKSPACE</div>", unsafe_allow_html=True)
    st.caption("XAI-StockGPT v2 · Local research environment")
    st.markdown("<div class='note'>Research interface only. Model outputs are experimental and are not investment advice.</div>", unsafe_allow_html=True)

st.markdown(f"<div class='hero'><div class='eyebrow'>EXPLAINABLE AI · EQUITY RESEARCH</div><h1>{page if page != 'Overview' else 'Market intelligence, made explainable.'}</h1><p>Explore model performance, market data and the features driving machine-learning decisions.</p></div>", unsafe_allow_html=True)

if page == "Overview":
    xgb = get_metrics("xgboost")
    xgb_test = xgb.get("test", {})
    models_ready = sum((ROOT / "models" / "prediction" / m / "metrics.json").exists() for m in MODELS)
    feature_count = len(get_xai("xgboost"))
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Models available", f"{models_ready}/4", "Saved model evaluation files")
    with c2: metric_card("XGBoost test accuracy", f"{xgb_test.get('accuracy',0)*100:.2f}%" if xgb_test else "—", "Locked test-set result")
    with c3: metric_card("Features explained", str(feature_count), "Global feature importance")
    with c4: metric_card("Research status", "READY" if models_ready else "SETUP", "Local artifacts detected")
    st.markdown("<div class='section-title'>Model performance snapshot</div><div class='section-sub'>Metrics below are from the project's saved locked test-set evaluation, not live trading predictions.</div>", unsafe_allow_html=True)
    rows=[]
    for m in MODELS:
        test=get_metrics(m).get("test",{})
        if test: rows.append({"Model":MODEL_LABELS[m],"Accuracy":test.get("accuracy",0),"Balanced accuracy":test.get("balanced_accuracy",0),"Macro F1":test.get("macro_f1",0),"MCC":test.get("mcc",0)})
    if rows:
        df=pd.DataFrame(rows).sort_values("Accuracy",ascending=False)
        left,right=st.columns([1.35,1])
        with left:
            fig=px.bar(df,x="Model",y="Accuracy",text=df["Accuracy"].map(lambda v:f"{v:.1%}"),color="Model",color_discrete_sequence=["#5ed8ed","#4adea3","#9b8cff","#e7b66c"])
            fig.update_traces(textposition="outside",showlegend=False)
            st.plotly_chart(chart_layout(fig,320),use_container_width=True)
        with right:
            st.dataframe(df.style.format({"Accuracy":"{:.2%}","Balanced accuracy":"{:.2%}","Macro F1":"{:.2%}","MCC":"{:.3f}"}),hide_index=True,use_container_width=True)
    else:
        st.info("No saved model metrics found yet. Run the pipeline from Pipeline Control.")
    st.markdown("<div class='section-title'>How the system works</div>", unsafe_allow_html=True)
    steps=st.columns(5)
    for col,(num,title,desc) in zip(steps,[("01","Market data","Collect & validate"),("02","Features","Technical indicators"),("03","Prediction","Train ML models"),("04","Explain","SHAP attribution"),("05","Language","Finance LLM rationale")]):
        with col: st.markdown(f"<div class='metric' style='min-height:105px'><div class='eyebrow'>{num}</div><b>{title}</b><div class='sub'>{desc}</div></div>",unsafe_allow_html=True)

elif page == "Market Explorer":
    left,right=st.columns([1,3])
    with left:
        company=st.selectbox("Select company",list(TICKERS.keys()))
        period=st.selectbox("Historical range",["1mo","3mo","6mo","1y","2y","5y"],index=3)
        st.caption("Market prices are fetched from Yahoo Finance when yfinance and an internet connection are available.")
    ticker=TICKERS[company]
    try:
        import yfinance as yf
        hist=yf.Ticker(ticker).history(period=period,auto_adjust=True)
        with right:
            if hist.empty: st.warning("No market data returned for this symbol. Check your internet connection or try again later.")
            else:
                latest=float(hist.Close.iloc[-1]); previous=float(hist.Close.iloc[-2]) if len(hist)>1 else latest
                change=(latest/previous-1)*100 if previous else 0
                a,b,c=st.columns(3)
                with a: metric_card("Latest close",f"₹{latest:,.2f}",ticker)
                with b: metric_card("Daily change",f"{change:+.2f}%","Versus previous close")
                with c: metric_card("Period high",f"₹{hist.High.max():,.2f}",f"Period low ₹{hist.Low.min():,.2f}")
                fig=go.Figure(); fig.add_trace(go.Candlestick(x=hist.index,open=hist.Open,high=hist.High,low=hist.Low,close=hist.Close,name=company,increasing_line_color="#4adea3",decreasing_line_color="#f07883"))
                fig.update_layout(xaxis_rangeslider_visible=False)
                st.markdown("<div class='section-title'>Price history</div>",unsafe_allow_html=True)
                st.plotly_chart(chart_layout(fig,440),use_container_width=True)
                vol=px.bar(x=hist.index,y=hist.Volume,labels={"x":"Date","y":"Volume"})
                st.plotly_chart(chart_layout(vol,180),use_container_width=True)
    except Exception as e:
        st.error(f"Could not load market data: {e}")
    st.markdown("<div class='note'>Historical market chart only. This screen does not currently generate a fresh per-stock BUY / HOLD / SELL prediction; the saved models are evaluation artifacts.</div>",unsafe_allow_html=True)

elif page == "Explainability":
    model=st.selectbox("Prediction model",MODELS,format_func=lambda x:MODEL_LABELS[x])
    items=get_xai(model)
    if items:
        df=pd.DataFrame(items).sort_values("importance",ascending=True).tail(20)
        fig=px.bar(df,x="importance",y="feature",orientation="h",color="importance",color_continuous_scale=["#24445a","#4adea3"])
        fig.update_layout(coloraxis_showscale=False)
        st.markdown("<div class='section-title'>Global feature importance</div><div class='section-sub'>These are model-level importance values saved by the research pipeline, not a per-stock SHAP explanation.</div>",unsafe_allow_html=True)
        st.plotly_chart(chart_layout(fig,560),use_container_width=True)
        top=sorted(items,key=lambda x:x.get("importance",0),reverse=True)[:5]
        st.markdown("<div class='section-title'>What the model relies on most</div>",unsafe_allow_html=True)
        cols=st.columns(len(top))
        for col,item in zip(cols,top):
            with col: metric_card(item["feature"].replace("_"," ").title(),f"{item['importance']:.3f}","Global importance")
    else: st.info("No feature importance artifact found for this model. Run the SHAP explanation stage.")
    faith=load_json(ROOT/"data"/"explanations"/model/"faithfulness_metrics.json",{})
    if faith:
        st.markdown("<div class='section-title'>Faithfulness check</div>",unsafe_allow_html=True)
        a,b,c=st.columns(3)
        with a: metric_card("Baseline confidence",f"{faith.get('baseline_confidence',0):.3f}")
        with b: metric_card("Top feature",faith.get("top_feature_perturbed","—"))
        with c: metric_card("Confidence drop",f"{faith.get('top_confidence_drop',0):.3f}")
        st.caption("Perturbation results are reported as stored in the project's faithfulness metrics.")

elif page == "Model Lab":
    st.markdown("<div class='section-title'>Compare trained models</div><div class='section-sub'>Evaluation metrics from each model's saved test split.</div>",unsafe_allow_html=True)
    rows=[]
    for m in MODELS:
        data=get_metrics(m).get("test",{})
        if data: rows.append({"Model":MODEL_LABELS[m],**{k:data.get(k) for k in ["accuracy","balanced_accuracy","macro_f1","weighted_f1","mcc"]}})
    if rows:
        df=pd.DataFrame(rows)
        st.dataframe(df.style.format({c:"{:.2%}" for c in ["accuracy","balanced_accuracy","macro_f1","weighted_f1"]}|{"mcc":"{:.3f}"}),hide_index=True,use_container_width=True)
        selected=st.selectbox("Inspect model",[r["Model"] for r in rows])
        key=next(k for k,v in MODEL_LABELS.items() if v==selected)
        test=get_metrics(key).get("test",{})
        cm=test.get("confusion_matrix")
        if cm:
            fig=px.imshow(cm,x=["Predicted SELL","Predicted HOLD","Predicted BUY"],y=["Actual SELL","Actual HOLD","Actual BUY"],text_auto=True,color_continuous_scale="Blues",aspect="auto")
            st.plotly_chart(chart_layout(fig,360),use_container_width=True)
    else: st.info("Train a model to populate this comparison.")
    st.markdown("<div class='note'>The archived benchmark report shows roughly 40–41% accuracy for the listed classifiers. Treat this as an experimental research result, not evidence of a profitable strategy.</div>",unsafe_allow_html=True)

elif page == "Pipeline Control":
    st.markdown("<div class='section-title'>Run research pipeline stages</div><div class='section-sub'>Each action runs the existing Python script from the project root and prints its output here.</div>",unsafe_allow_html=True)
    st.markdown("<div class='note'>Run stages in order: collect data → build features → train model → generate explanations → benchmark. These jobs can take time and download data from the internet.</div>",unsafe_allow_html=True)
    st.write("")
    with st.expander("1 · Collect and validate market data",expanded=True):
        symbols=st.text_input("Ticker symbols (space separated)","RELIANCE.NS TCS.NS INFY.NS HDFCBANK.NS SBIN.NS")
        if st.button("Run data collection",key="collect"):
            run_script(["scripts/collect_data.py","--symbols",*symbols.split()],"Data collection")
    with st.expander("2 · Build features and targets"):
        if st.button("Build features",key="features"): run_script(["scripts/build_features.py"],"Feature engineering")
    with st.expander("3 · Train a prediction model"):
        m=st.selectbox("Model to train",MODELS,format_func=lambda x:MODEL_LABELS[x],key="train_model")
        if st.button("Train selected model",key="train"): run_script(["scripts/train.py","--model",m],f"{MODEL_LABELS[m]} training")
    with st.expander("4 · Generate SHAP explanations"):
        m2=st.selectbox("Model to explain",MODELS,format_func=lambda x:MODEL_LABELS[x],key="explain_model")
        if st.button("Generate explanations",key="explain"): run_script(["scripts/explain.py","--model",m2],f"{MODEL_LABELS[m2]} explanations")
    with st.expander("5 · Run benchmark"):
        if st.button("Run evaluation benchmark",key="benchmark"): run_script(["scripts/benchmark.py"],"Benchmark")
    st.caption(f"Project root: {ROOT}")

st.markdown("<div style='border-top:1px solid #202d40;margin-top:34px;padding-top:14px;color:#65758b;font-size:11px'>STOCKGPT · EXPLAINABLE AI RESEARCH WORKBENCH · NOT FINANCIAL ADVICE</div>",unsafe_allow_html=True)
