import streamlit as st
import requests
import pandas as pd
from io import StringIO

st.set_page_config(page_title="Exo-Planet Hunters: Intelligent Exoplanet Classification Lab", layout="wide")

# ----- Custom CSS for App Branding & Buttons -----
st.markdown(
    """
    <style>
    .big-title {
        font-size:3.6rem!important;
        font-weight:900;
        color:#ffb300;
        letter-spacing:1.5px;
        margin-bottom:.45em;
    }
    .subtitle {
        font-size:1.32rem!important;
        font-weight:400;
        color:#7fd1ff;
        margin-bottom:.45em;
    }
    .section {
        font-size:1.15rem!important;
        font-weight:700;
        color:#2ec4b6;
        margin:1em 0 .5em 0;
    }
    .results {
        background-color:#202c39;
        border-radius:12px;
        padding:17px 25px 12px 25px;
        margin-bottom:1em;
        color:#ffe082;
        font-size:1.1rem!important;
    }
    .footer {
        color:#adadad;
        font-size: 1.09rem;
        margin-top:.7em;
    }
    .leaderboard-title {
        color:#ffe082;
        padding:10px 0 0 0;
        font-size:1.2rem!important;
        font-weight:800;
        letter-spacing:1px;
    }
    .feedback-link-box {
        border: 2px solid #00ff5e;
        background: rgba(40, 45, 80, 0.53);
        padding: 16px;
        border-radius: 10px;
        margin-top:18px;
        margin-bottom:24px;
        text-align:center;
        font-size:1.21rem;
        font-weight:650;
        color:#43db25;
    }
    a.feedback-link {
        color: #2980f0;
        font-weight:800;
        text-decoration: underline;
        font-size: 1.21rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="big-title">🚀 ExoHunter: Intelligent Exoplanet Classification Lab</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Automated ML • Downloadable Reports • Instant Interpretation • Judge and User Friendly</div>', unsafe_allow_html=True)

if "run_history" not in st.session_state:
    st.session_state['run_history'] = []

st.markdown('<div class="section">Step 1: Upload your Kepler exoplanet CSV ⬇️</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["csv"], help="Drag or browse a .csv file, 200MB max.")

user_email = st.text_input("Optional: Enter your email for report delivery or log tracking", placeholder="you@domain.com")

if uploaded_file is not None:
    st.markdown('<div class="section">Step 2: Click below to Analyze</div>', unsafe_allow_html=True)
    if st.button("⭐ Analyze My File & Show Results", use_container_width=True, type="primary"):
        webhook_url = "https://hook.eu2.make.com/rcjsz0pgphpfqxward65cp8y2pxp9lpv"
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        data = {"email": user_email}

        with st.spinner("🔭 Analyzing with ExoHunter AI..."):
            response = requests.post(webhook_url, files=files, data=data)

        if response.status_code == 200:
            try:
                result = response.json()
                this_run = {
                    "filename": uploaded_file.name,
                    "accuracy": round(result.get("accuracy", 0.0), 4),
                    "details": result
                }
                st.session_state.run_history = ([this_run] + st.session_state.run_history)[:5]

                # --- Results Section ----
                st.markdown('<div class="section">Results Overview</div>', unsafe_allow_html=True)
                rounded_acc = f"{result.get('accuracy', 'N/A'):.3f}" if isinstance(result.get("accuracy"), float) else result.get("accuracy")
                st.markdown(f"<div class='results'>🎯 <b>Model Accuracy:</b> {rounded_acc}</div>", unsafe_allow_html=True)

                # Plots, if present
                if result.get("image_url"):
                    st.image(result["image_url"], caption="Confusion Matrix")
                if result.get("feature_importances_url"):
                    st.image(result["feature_importances_url"], caption="Feature Importances")
                if result.get("roc_curve_url"):
                    st.image(result["roc_curve_url"], caption="ROC Curve")
                else:
                    st.info("ROC curve only appears for binary classification cases.")

                # Dynamic data table & download predictions
                pred_csv_string = result.get("preview_csv")
                if pred_csv_string:
                    sampledf = pd.read_csv(StringIO(pred_csv_string))
                    st.markdown("#### 🔬 Sample Predictions (scrollable) (Top 50 rows)")
                    st.dataframe(sampledf.head(50), use_container_width=True)
                    csv_buf = StringIO()
                    sampledf.to_csv(csv_buf, index=False)
                    st.download_button(
                        label="⬇️ Download All Predictions (CSV)",
                        data=csv_buf.getvalue(),
                        file_name='model_predictions.csv',
                        mime='text/csv',
                        use_container_width=True,
                        help="Download the complete predictions file for your run."
                    )
                else:
                    st.info("No sample predictions CSV returned for preview.")

                # PDF report
                if "pdf_url" in result and result["pdf_url"]:
                    st.markdown(
                        f"<br>📄 <b>Download your detailed PDF report:</b> "
                        f"<a href='{result['pdf_url']}' target='_blank'>Click here</a>",
                        unsafe_allow_html=True
                    )

                # --- Google Form Feedback Section ---
                st.markdown(
                    f"""
                    <div class='feedback-link-box'>
                    <span>📝 <b>Judges & Users:</b> Please rate and review this result using our <a class='feedback-link' href="https://forms.gle/ddMBDv5rLJYtMJbb9" target="_blank">Official Feedback Form</a>.<br>
                    <span style='color:#dddddd; font-size:.98em;'>Your review helps the model to improve!</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"Error displaying results: {e}")
        else:
            st.error("❌ There was a problem with the analysis. Please double-check your CSV and try again.")

    # ====== LEADERBOARD (Session-based) ======
    if len(st.session_state['run_history']) > 0:
        st.markdown("<div class='leaderboard-title'>🏅 Leaderboard: Recent Runs</div>", unsafe_allow_html=True)
        hist_df = pd.DataFrame([
            {"Filename": x["filename"], "Accuracy": x["accuracy"]}
            for x in st.session_state['run_history']
        ])
        st.dataframe(hist_df, use_container_width=True)

else:
    st.info("Please upload your dataset to get started. Only CSV files supported.")

st.markdown(
    """
    <hr style='border:1px solid #FFD700; margin-top:32px;'>
    <div class='footer'>
    Built by Team Exo-Planet Hunters · NASA Space Apps 2025 <br>
    <i>Kepler data, open science & ML for a brighter universe!</i> <br>
    <i>Team members: Yadav Prachi SandipKumar, Mankad Saniya Anvarbhai.</i>

    </div>
    """,
    unsafe_allow_html=True
)
