import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Model Library
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except ImportError:
    st.error("Please install XGBoost in the terminal first: pip install xgboost")
    st.stop()
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import shap

# ==========================================
# Page & Global Font Configuration
# ==========================================
st.set_page_config(page_title="Pneumoconiosis Risk Prediction System", layout="wide")

st.markdown("""
<style>
    * {
        font-family: Arial, Helvetica, 'DejaVu Sans', sans-serif !important;
    }
    div[data-testid="stNumberInput"] label p,
    div[data-testid="stSelectbox"] label p {
        font-size: 20px !important;
        font-weight: bold !important;
    }
    [data-testid="stMetricValue"] { font-family: Arial, Helvetica, 'DejaVu Sans', sans-serif !important; }
    input[type="number"] {
        font-family: Arial, Helvetica, 'DejaVu Sans', sans-serif !important;
        font-size: 30px !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""<h1 style='font-family: Arial, Helvetica, 'DejaVu Sans', sans-serif; font-size: 54px; font-weight: 700; white-space: nowrap; margin: 0 0 0.2rem 0; line-height: 1.05;'>Miner Pneumoconiosis Risk Prediction System</h1>""", unsafe_allow_html=True)
st.markdown("Enter feature data to predict the risk of pneumoconiosis and perform interpretability analysis.")
st.divider()

COLOR_BLUE = "#316395"
COLOR_RED = "#B82E2E"


# ==========================================
# 工具函数：渲染带 CSS 颜色劫持的 JS 力图（保留备用）
# ==========================================
def st_shap(plot, height=300, min_width=1800):
    shap_html = f"""
    <head>
        {shap.getjs()}
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                background: white;
                font-family: Arial, Helvetica, 'DejaVu Sans', sans-serif !important;
            }}
            .shap-wrap {{
                width: 100%;
                overflow-x: auto;
                overflow-y: hidden;
                padding: 6px 0 0 0;
                background: white;
            }}
            .shap-inner {{
                min-width: {min_width}px;
                width: {min_width}px;
                padding-right: 12px;
            }}
            .shap-inner svg {{
                overflow: visible;
            }}
            body, div, span, text, g {{
                font-family: Arial, Helvetica, 'DejaVu Sans', sans-serif !important;
            }}
            path[fill="#ff0052"] {{ fill: {COLOR_RED} !important; }}
            path[stroke="#ff0052"] {{ stroke: {COLOR_RED} !important; }}
            text[fill="#ff0052"] {{ fill: {COLOR_RED} !important; }}
            path[fill="#008bfb"] {{ fill: {COLOR_BLUE} !important; }}
            path[stroke="#008bfb"] {{ stroke: {COLOR_BLUE} !important; }}
            text[fill="#008bfb"] {{ fill: {COLOR_BLUE} !important; }}
        </style>
    </head>
    <body>
        <div class="shap-wrap">
            <div class="shap-inner">{plot.html()}</div>
        </div>
    </body>
    """
    components.html(shap_html, height=height, scrolling=False)


# Matplotlib 基础配置（专供图形使用）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans'] + plt.rcParams['font.sans-serif']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.default'] = 'regular'


# ==========================================
# Core Logic & 5-Stack Model Loading
# ==========================================
@st.cache_resource
def load_and_train_model():
    file_path = '煤矿数据.xlsx'
    features = ['years', 'time/week', 'blasting', 'transport', 'extract', 'support', 'repair', 'other', 'max', 'Ctwa',
                'SiO2', 'protect']

    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        df = pd.DataFrame(np.random.rand(200, 13), columns=features + ['abnormal'])
        df['years'] = df['years'] * 30
        df['SiO2'] = df['SiO2'] * 100

    x = df[features].copy()
    y = df['abnormal'].copy()
    x = x.fillna(x.median(numeric_only=True))

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    base_models = [
        RandomForestRegressor(n_estimators=50, max_depth=8, min_samples_leaf=6, random_state=100),
        RidgeCV(alphas=[0.1, 1, 10, 50]),
        KNeighborsRegressor(n_neighbors=5),
        SVR(kernel="rbf", C=10, gamma=0.1),
        XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=100, n_jobs=-1)
    ]

    kf = KFold(n_splits=5, shuffle=True, random_state=100)
    xtrain_meta = np.zeros((x_scaled.shape[0], len(base_models)))

    for m, model in enumerate(base_models):
        for train_idx, val_idx in kf.split(x_scaled):
            y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            model.fit(x_scaled[train_idx], y_train_fold)
            xtrain_meta[val_idx, m] = model.predict(x_scaled[val_idx])
        model.fit(x_scaled, y)

    meta_model = RidgeCV()
    meta_model.fit(xtrain_meta, y)
    background = shap.sample(pd.DataFrame(x_scaled, columns=features), min(200, len(x_scaled)), random_state=42)

    def stacking_predict(X_input):
        base_preds = [model.predict(X_input) for model in base_models]
        return meta_model.predict(np.column_stack(base_preds))

    explainer = shap.KernelExplainer(stacking_predict, background)
    return base_models, meta_model, scaler, explainer, features, stacking_predict


with st.spinner("Loading model and data, please wait..."):
    base_models, meta_model, scaler, explainer, features, stacking_predict = load_and_train_model()

# ==========================================
# Part 1: Input Section
# ==========================================
st.header("Step 1: Input Prediction Data")

input_data = {}
col1, col2 = st.columns(2)
with col1:
    input_data['years'] = st.number_input("Years of Service", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
    input_data['max'] = st.number_input("Peak Dust Concentration", value=5.0)
    input_data['SiO2'] = st.number_input("Free Silica Content", value=10.0)
with col2:
    input_data['time/week'] = st.number_input("Weekly Work Hours", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
    input_data['Ctwa'] = st.number_input("Time-Weighted Average Concentration", value=2.0)
    input_data['protect'] = st.number_input("Effectiveness of Protective Measures", value=1.0)

st.subheader("Job Type Selection")
job_options = {
    "Blasting": "blasting",
    "Transport": "transport",
    "Extraction": "extract",
    "Support": "support",
    "Repair": "repair",
    "Other": "other"
}
selected_job_zh = st.selectbox("Please select your job type:", list(job_options.keys()))
selected_job_en = job_options[selected_job_zh]
for en_name in job_options.values():
    input_data[en_name] = 1.0 if en_name == selected_job_en else 0.0

input_df = pd.DataFrame([input_data])[features]

st.markdown("<br>", unsafe_allow_html=True)

display_names = {
    'years': 'Years', 'time/week': 'Time/Week', 'blasting': 'Blasting',
    'transport': 'Transport', 'extract': 'Extract', 'support': 'Support',
    'repair': 'Repair', 'other': 'Other', 'max': 'Max',
    'Ctwa': r'C$_{twa}$', 'SiO2': 'SiO₂', 'protect': 'Protect'
}

# ==========================================
# Part 2: Prediction & Analysis
# ==========================================
if st.button("Run Prediction", type="primary", use_container_width=True):
    input_scaled = scaler.transform(input_df)
    prediction = stacking_predict(input_scaled)[0]

    if prediction <= 5:
        risk_level, risk_color = "Low Risk", "🟢"
    elif prediction <= 12:
        risk_level, risk_color = "Medium Risk", "🟡"
    else:
        risk_level, risk_color = "High Risk", "🔴"

    st.divider()
    st.header("Step 2: Prediction Results & Analysis")

    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric(label="Pneumoconiosis Risk Level", value=f"{risk_color} {risk_level}")
    with col_res2:
        st.metric(label="Model Predicted Abnormal Risk Value", value=f"{prediction:.4f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Click to view risk attribution charts and rectification suggestions", expanded=True):
        with st.spinner("Generating analysis charts..."):
            shap_values_raw = explainer.shap_values(input_scaled, nsamples=100)
            unselected_jobs = [job for job in job_options.values() if job != selected_job_en]
            keep_indices = [i for i, f in enumerate(features) if f not in unselected_jobs]
            unselected_indices = [i for i, f in enumerate(features) if f in unselected_jobs]
            unselected_shap_sum = np.sum(shap_values_raw[0][unselected_indices])
            expected_val = explainer.expected_value
            if isinstance(expected_val, (list, np.ndarray)):
                expected_val = expected_val[0]

            adjusted_base_value = float(expected_val) + float(unselected_shap_sum)
            adjusted_values = shap_values_raw[0][keep_indices]
            adjusted_data = input_df.iloc[0].values[keep_indices]
            adjusted_features_display = [display_names[features[i]] for i in keep_indices]

            shap_exp = shap.Explanation(
                values=adjusted_values,
                base_values=adjusted_base_value,
                data=adjusted_data,
                feature_names=adjusted_features_display
            )

            try:
                from shap.plots import colors as shap_colors
                shap_colors.red.rgb = mcolors.hex2color(COLOR_RED)
                shap_colors.blue.rgb = mcolors.hex2color(COLOR_BLUE)
            except Exception:
                pass

            # --- 图 1：瀑布图 ---
            st.subheader("Risk Accumulation Attribution Analysis")
            fig_waterfall, ax_wf = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_exp, show=False, max_display=10)

            for patch in ax_wf.patches:
                try:
                    fc = mcolors.to_rgb(patch.get_facecolor())
                    if fc[0] > fc[2] + 0.1:
                        patch.set_facecolor(COLOR_RED)
                        patch.set_edgecolor(COLOR_RED)
                    elif fc[2] > fc[0] + 0.1:
                        patch.set_facecolor(COLOR_BLUE)
                        patch.set_edgecolor(COLOR_BLUE)
                except Exception:
                    pass

            for text in ax_wf.texts:
                try:
                    text.set_fontweight('bold')
                    c = mcolors.to_rgb(text.get_color())
                    if sum(c) > 2.8:
                        continue
                    if c[0] > c[2] + 0.1:
                        text.set_color(COLOR_RED)
                    elif c[2] > c[0] + 0.1:
                        text.set_color(COLOR_BLUE)
                except Exception:
                    pass

            for line in ax_wf.lines:
                if line.get_linestyle() == '--':
                    line.set_color('#cccccc')

            plt.tight_layout()
            st.pyplot(fig_waterfall)
            plt.close(fig_waterfall)

            # --- 图 2：云端稳定版 JS 力图 ---
            st.markdown("---")
            st.subheader("Risk Driving Force Analysis")
            try:
                rounded_features = np.round(adjusted_data.astype(float), 2)
                force_plot_js = shap.force_plot(
                    base_value=float(adjusted_base_value),
                    shap_values=adjusted_values,
                    features=rounded_features,
                    feature_names=adjusted_features_display,
                    plot_cmap=[COLOR_BLUE, COLOR_RED],
                    contribution_threshold=0.0
                )
                min_width = max(1800, 260 * len(adjusted_features_display))
                st_shap(force_plot_js, height=320, min_width=min_width)
            except Exception as e:
                st.warning(f"Issue generating Force Plot: {e}")

        st.markdown("---")
        st.subheader("Targeted Dust Prevention & Rectification Suggestions")
        feature_shap_dict = {feat: val for feat, val in zip(adjusted_features_display, adjusted_values)}
        sorted_features = sorted(feature_shap_dict.items(), key=lambda x: x[1], reverse=True)
        top_risk_features = [item for item in sorted_features if item[1] > 0][:3]

        if top_risk_features:
            st.write(
                "Based on the model attribution analysis, the following factors are the core reasons for the current increased risk. It is recommended to focus on implementing the following rectification measures:")
            measures_dict = {
                'Max': "**Reduce Peak Dust Concentration (Max)**: Must install spray dust reduction at main dust generation points; optimize ventilation; wet operation.",
                'Ctwa': "**Control Time-Weighted Average Concentration (Ctwa)**: Improve ventilation and dust removal efficiency; fully enclose dust reduction in key areas; normalize the opening of water curtains.",
                r'C$_{twa}$': "**Control Time-Weighted Average Concentration (Ctwa)**: Improve ventilation and dust removal efficiency; fully enclose dust reduction in key areas; normalize the opening of water curtains.",
                'SiO₂': "**Handle High Free Silica (SiO₂)**: Adopt long-extraction and short-pressure combined dust removal fan scheme; wear the highest protection level masks for such positions.",
                'Years': "**High Service Years Health Management (Years)**: High cumulative risk. Increase frequency of physical examinations; prioritize off-dust job rotation.",
                'Time/Week': "**Optimize Weekly Work Hours (Time/Week)**: Strictly control operation hours; implement off-dust rest system.",
                'Protect': "**Improve Protective Measures Effectiveness (Protect)**: Inspect protective equipment; strengthen mask airtightness inspection and training.",
                'Blasting': "**Blasting Operation Standards**: Water stemming and water-sealed blasting technology; spray and wash before and after blasting; ensure ventilation and dust removal time.",
                'Extract': "**Extraction Operation Standards**: Ensure internal and external sprays of shearer meet standards; link support sprays; water injection into coal seam.",
                'Support': "**Support Operation Standards**: Strictly prohibit dry drilling, use wet rock drilling; use wet spraying for shotcrete operations.",
                'Transport': "**Transport Operation Standards**: Enclosed spray at transfer points; clean up accumulated dust to prevent secondary dust generation; keep belt transport wet.",
                'Repair': "**Repair Operation Standards**: Spray water to reduce dust before repairing in high concentration areas; equip with portable dust-proof respirators.",
                'Other': "**Comprehensive Position Protection**: Dust prevention measures adapted to local conditions; strengthen supervision of wearing personal protective equipment."
            }
            for idx, (feat, val) in enumerate(top_risk_features):
                st.info(f"**[Risk Factor {idx + 1}] (Risk Increment: +{val:.2f})** \n\n {measures_dict.get(feat, '')}")
