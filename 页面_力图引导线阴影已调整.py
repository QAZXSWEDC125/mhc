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
        font-family: 'Times New Roman', serif !important;
    }
    div[data-testid="stNumberInput"] label p,
    div[data-testid="stSelectbox"] label p {
        font-size: 20px !important;
        font-weight: bold !important;
    }
    [data-testid="stMetricValue"] { font-family: 'Times New Roman', serif !important; }
    input[type="number"] {
        font-family: 'Times New Roman', serif !important;
        font-size: 30px !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Miner Pneumoconiosis Risk Prediction System")
st.markdown("Enter feature data to predict the risk of pneumoconiosis and perform interpretability analysis.")
st.divider()

COLOR_BLUE = "#316395"
COLOR_RED = "#B82E2E"


# ==========================================
# 工具函数：渲染带 CSS 颜色劫持的 JS 力图（保留备用）
# ==========================================
def st_shap(plot, height=200):
    shap_html = f"""
    <head>
        {shap.getjs()}
        <style>
            body, div, span, text, g {{
                font-family: 'Times New Roman', serif !important;
            }}
            path[fill="#ff0052"] {{ fill: {COLOR_RED} !important; }}
            path[stroke="#ff0052"] {{ stroke: {COLOR_RED} !important; }}
            text[fill="#ff0052"] {{ fill: {COLOR_RED} !important; }}
            path[fill="#008bfb"] {{ fill: {COLOR_BLUE} !important; }}
            path[stroke="#008bfb"] {{ stroke: {COLOR_BLUE} !important; }}
            text[fill="#008bfb"] {{ fill: {COLOR_BLUE} !important; }}
        </style>
    </head>
    <body>{plot.html()}</body>
    """
    components.html(shap_html, height=height)


# Matplotlib 基础配置（专供图形使用）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
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

            # --- 图 2：静态 Matplotlib 力图 ---
            st.markdown("---")
            st.subheader("Risk Driving Force Analysis")
            try:
                rounded_features = np.round(adjusted_data.astype(float), 2)

                plt.figure(figsize=(20, 4), dpi=800)
                shap.force_plot(
                    base_value=float(adjusted_base_value),
                    shap_values=adjusted_values,
                    features=rounded_features,
                    feature_names=adjusted_features_display,
                    matplotlib=True,
                    show=False,
                    figsize=(20, 4),
                    text_rotation=0,
                    contribution_threshold=0.0
                )

                fig_force = plt.gcf()
                if fig_force.axes:
                    ax_force = fig_force.axes[0]

                    def _rgb_close(color_a, color_b, tol=0.08):
                        try:
                            a = np.array(mcolors.to_rgb(color_a), dtype=float)
                            b = np.array(mcolors.to_rgb(color_b), dtype=float)
                            return np.max(np.abs(a - b)) <= tol
                        except Exception:
                            return False

                    def _poly_center_x(patch):
                        try:
                            verts = np.asarray(patch.get_xy(), dtype=float)
                            if verts.ndim == 2 and verts.shape[1] == 2:
                                return float(np.mean(verts[:, 0]))
                        except Exception:
                            pass
                        return None

                    def _data_width_of_text(txt_obj, renderer):
                        try:
                            bbox = txt_obj.get_window_extent(renderer=renderer)
                            x0 = ax_force.transData.inverted().transform((bbox.x0, bbox.y0))[0]
                            x1 = ax_force.transData.inverted().transform((bbox.x1, bbox.y0))[0]
                            return abs(float(x1 - x0))
                        except Exception:
                            return 0.8

                    DEFAULT_BLUE = "#1E88E5"
                    DEFAULT_RED = "#FF0D57"
                    LIGHT_BLUE = "#D1E6FA"
                    LIGHT_RED = "#FBB0C8"

                    # 1) 力图主色条改为与瀑布图一致，并保留白色分隔纹理
                    for patch in ax_force.patches:
                        try:
                            fc = patch.get_facecolor()
                            ec = patch.get_edgecolor()
                            fc_alpha = float(fc[3]) if fc is not None and len(fc) >= 4 else 0.0
                            ec_alpha = float(ec[3]) if ec is not None and len(ec) >= 4 else 0.0

                            if fc_alpha > 0.5:
                                if _rgb_close(fc, DEFAULT_BLUE):
                                    patch.set_facecolor(COLOR_BLUE)
                                    patch.set_edgecolor((0, 0, 0, 0))
                                elif _rgb_close(fc, DEFAULT_RED):
                                    patch.set_facecolor(COLOR_RED)
                                    patch.set_edgecolor((0, 0, 0, 0))
                            elif fc_alpha < 0.1 and ec_alpha > 0:
                                if _rgb_close(ec, LIGHT_BLUE) or _rgb_close(ec, DEFAULT_BLUE):
                                    patch.set_facecolor((1, 1, 1, 0))
                                    patch.set_edgecolor('#EAF3FB')
                                    patch.set_linewidth(1.8)
                                    patch.set_zorder(6)
                                elif _rgb_close(ec, LIGHT_RED) or _rgb_close(ec, DEFAULT_RED):
                                    patch.set_facecolor((1, 1, 1, 0))
                                    patch.set_edgecolor('#FBE5EE')
                                    patch.set_linewidth(1.8)
                                    patch.set_zorder(6)
                        except Exception:
                            pass

                    def _nearest_main_color(x_mid):
                        candidates = []
                        for patch in ax_force.patches:
                            try:
                                fc = patch.get_facecolor()
                                fc_alpha = float(fc[3]) if fc is not None and len(fc) >= 4 else 0.0
                                if fc_alpha <= 0.5:
                                    continue
                                cx = _poly_center_x(patch)
                                if cx is None:
                                    continue
                                candidates.append((abs(cx - x_mid), fc))
                            except Exception:
                                pass
                        if not candidates:
                            return COLOR_BLUE
                        _, fc = sorted(candidates, key=lambda z: z[0])[0]
                        if _rgb_close(fc, COLOR_RED, tol=0.15):
                            return COLOR_RED
                        return COLOR_BLUE

                    # 2) 下方阴影颜色与对应色条保持一致
                    for idx, img in enumerate(ax_force.images):
                        try:
                            ex = img.get_extent()
                            x_mid = float((float(ex[0]) + float(ex[1])) / 2.0)
                            side_color = _nearest_main_color(x_mid)
                            shadow_cmap = mcolors.LinearSegmentedColormap.from_list(
                                f"shadow_{idx}",
                                [(1.0, 1.0, 1.0, 0.0), mcolors.to_rgba(side_color, 0.22)]
                            )
                            img.set_cmap(shadow_cmap)
                            img.set_alpha(1.0)
                            img.set_zorder(0.5)
                        except Exception:
                            pass

                    # 3) 引导线颜色改为与力图对应色完全一致
                    for line in ax_force.lines:
                        try:
                            c = line.get_color()
                            if _rgb_close(c, DEFAULT_BLUE) or _rgb_close(c, LIGHT_BLUE) or _rgb_close(c, COLOR_BLUE):
                                line.set_color(COLOR_BLUE)
                                line.set_linewidth(max(line.get_linewidth(), 1.6))
                            elif _rgb_close(c, DEFAULT_RED) or _rgb_close(c, LIGHT_RED) or _rgb_close(c, COLOR_RED):
                                line.set_color(COLOR_RED)
                                line.set_linewidth(max(line.get_linewidth(), 1.6))
                        except Exception:
                            pass

                    # 4) 标签颜色与力图主色一致
                    for txt in ax_force.texts:
                        try:
                            c = txt.get_color()
                            if _rgb_close(c, DEFAULT_BLUE) or _rgb_close(c, LIGHT_BLUE) or _rgb_close(c, COLOR_BLUE):
                                txt.set_color(COLOR_BLUE)
                            elif _rgb_close(c, DEFAULT_RED) or _rgb_close(c, LIGHT_RED) or _rgb_close(c, COLOR_RED):
                                txt.set_color(COLOR_RED)
                        except Exception:
                            pass

                    # 5) 增加主色条高度，但不破坏白色分隔纹理
                    for patch in ax_force.patches:
                        try:
                            fc = patch.get_facecolor()
                            fc_alpha = float(fc[3]) if fc is not None and len(fc) >= 4 else 0.0
                            if fc_alpha > 0.5 and hasattr(patch, 'get_xy'):
                                verts = np.asarray(patch.get_xy(), dtype=float)
                                if verts.ndim == 2 and verts.shape[1] == 2:
                                    y = verts[:, 1].copy()
                                    positive = y > 0
                                    y[positive] = y[positive] * 1.28
                                    verts[:, 1] = y
                                    patch.set_xy(verts)
                                    patch.set_zorder(2)
                        except Exception:
                            pass

                    for patch in ax_force.patches:
                        try:
                            fc = patch.get_facecolor()
                            ec = patch.get_edgecolor()
                            fc_alpha = float(fc[3]) if fc is not None and len(fc) >= 4 else 0.0
                            ec_alpha = float(ec[3]) if ec is not None and len(ec) >= 4 else 0.0
                            if fc_alpha < 0.1 and ec_alpha > 0 and hasattr(patch, 'get_xy'):
                                verts = np.asarray(patch.get_xy(), dtype=float)
                                if verts.ndim == 2 and verts.shape[1] == 2:
                                    y = verts[:, 1].copy()
                                    positive = y > 0
                                    y[positive] = y[positive] * 1.28
                                    verts[:, 1] = y
                                    patch.set_xy(verts)
                                    patch.set_zorder(6)
                        except Exception:
                            pass

                    fig_force.canvas.draw()
                    renderer = fig_force.canvas.get_renderer()

                    # 6) 横向拉开底部标签，使引导线横向长度增加
                    label_texts = []
                    for txt in ax_force.texts:
                        try:
                            x, y = txt.get_position()
                            if y < 0 and "=" in txt.get_text():
                                label_texts.append(txt)
                        except Exception:
                            pass

                    def _nearest_label_line(txt_obj, old_x, old_y):
                        txt_color = txt_obj.get_color()
                        candidates = []
                        for line in ax_force.lines:
                            try:
                                lc = line.get_color()
                                xs = np.asarray(line.get_xdata(), dtype=float)
                                ys = np.asarray(line.get_ydata(), dtype=float)
                                if xs.size == 0 or ys.size == 0:
                                    continue
                                if not _rgb_close(lc, txt_color, tol=0.16):
                                    continue
                                score = abs(float(xs[-1]) - old_x) + 1.5 * abs(float(ys[-1]) - (old_y - 0.03))
                                candidates.append((score, line))
                            except Exception:
                                pass
                        return sorted(candidates, key=lambda z: z[0])[0][1] if candidates else None

                    def _move_label_and_line(txt_obj, new_x):
                        old_x, old_y = txt_obj.get_position()
                        txt_obj.set_position((new_x, old_y))
                        line = _nearest_label_line(txt_obj, old_x, old_y)
                        if line is not None:
                            try:
                                xs = np.asarray(line.get_xdata(), dtype=float)
                                ys = np.asarray(line.get_ydata(), dtype=float)
                                x0 = float(xs[0]) if xs.size else old_x
                                y_end = old_y - 0.03
                                if xs.size >= 3:
                                    line.set_xdata([x0, new_x, new_x])
                                    line.set_ydata([0.0, -0.08, y_end])
                                else:
                                    line.set_xdata([x0, new_x, new_x])
                                    line.set_ydata([0.0, -0.08, y_end])
                            except Exception:
                                pass

                    red_labels = [t for t in label_texts if _rgb_close(t.get_color(), COLOR_RED, tol=0.12)]
                    blue_labels = [t for t in label_texts if _rgb_close(t.get_color(), COLOR_BLUE, tol=0.12)]

                    def _spread_side(text_list, side='left'):
                        if not text_list:
                            return
                        if side == 'left':
                            ordered = sorted(text_list, key=lambda t: t.get_position()[0], reverse=True)
                        else:
                            ordered = sorted(text_list, key=lambda t: t.get_position()[0])

                        cursor = None
                        outward_bias = 0.20
                        for txt in ordered:
                            x, y = txt.get_position()
                            width_data = _data_width_of_text(txt, renderer)
                            gap = max(0.48, width_data * 0.16)
                            if cursor is None:
                                new_x = x - outward_bias if side == 'left' else x + outward_bias
                            else:
                                if side == 'left':
                                    new_x = min(x - outward_bias, cursor - gap)
                                else:
                                    new_x = max(x + outward_bias, cursor + gap)
                            _move_label_and_line(txt, new_x)
                            cursor = new_x

                    _spread_side(red_labels, side='left')
                    _spread_side(blue_labels, side='right')

                    # 7) 如仍有重叠，再做轻微错层下移，并同步调整引导线末端
                    fig_force.canvas.draw()
                    renderer = fig_force.canvas.get_renderer()
                    label_texts = sorted(
                        [t for t in ax_force.texts if "=" in t.get_text() and t.get_position()[1] < 0],
                        key=lambda t: t.get_window_extent(renderer=renderer).x0
                    )
                    moved_labels = []

                    for txt in label_texts:
                        bbox = txt.get_window_extent(renderer=renderer)
                        level = 0
                        for prev_bbox, prev_level in moved_labels:
                            overlap = bbox.x0 < (prev_bbox.x1 + 8)
                            close_row = abs(bbox.y0 - prev_bbox.y0) < 14
                            if overlap and close_row:
                                level = max(level, prev_level + 1)

                        if level > 0:
                            x, base_y = txt.get_position()
                            new_y = base_y - 0.07 * min(level, 2)
                            txt.set_position((x, new_y))
                            line = _nearest_label_line(txt, x, base_y)
                            if line is not None:
                                try:
                                    xs = np.asarray(line.get_xdata(), dtype=float)
                                    x0 = float(xs[0]) if xs.size else x
                                    line.set_xdata([x0, x, x])
                                    line.set_ydata([0.0, -0.08, new_y - 0.03])
                                except Exception:
                                    pass

                        fig_force.canvas.draw()
                        moved_labels.append((txt.get_window_extent(renderer=renderer), level))

                    # 8) 按更新后的图形元素重新收紧坐标范围
                    x_candidates = []
                    for patch in ax_force.patches:
                        try:
                            if hasattr(patch, 'get_xy'):
                                verts = np.asarray(patch.get_xy(), dtype=float)
                                if verts.ndim == 2 and verts.shape[1] >= 1:
                                    x_candidates.extend(verts[:, 0].tolist())
                        except Exception:
                            pass
                    for img in ax_force.images:
                        try:
                            ex = img.get_extent()
                            x_candidates.extend([float(ex[0]), float(ex[1])])
                        except Exception:
                            pass
                    for line in ax_force.lines:
                        try:
                            xs = np.asarray(line.get_xdata(), dtype=float)
                            xs = xs[np.isfinite(xs)]
                            if xs.size:
                                x_candidates.extend(xs.tolist())
                        except Exception:
                            pass
                    for txt in ax_force.texts:
                        try:
                            x, y = txt.get_position()
                            if np.isfinite(x) and y >= -0.34:
                                x_candidates.append(float(x))
                        except Exception:
                            pass

                    if x_candidates:
                        xmin = float(np.min(x_candidates))
                        xmax = float(np.max(x_candidates))
                        span = max(xmax - xmin, 1e-6)
                        left = xmin - max(span * 0.025, 0.12)
                        right = xmax + max(span * 0.02, 0.10)
                        if right <= left:
                            right = left + 1.0
                        ax_force.set_xlim(left, right)
                        ax_force.margins(x=0)

                    # 9) 字体与顶部说明微调
                    for ax in fig_force.axes:
                        ax.tick_params(labelsize=18, width=1.2)
                        for label in ax.get_xticklabels() + ax.get_yticklabels():
                            label.set_fontfamily('Times New Roman')
                            label.set_fontweight('bold')
                            label.set_fontsize(16)
                        for text_obj in ax.texts:
                            text_obj.set_fontfamily('Times New Roman')
                            text_obj.set_fontweight('bold')
                            txt_content = text_obj.get_text()
                            x_pos, y_pos = text_obj.get_position()
                            if '=' in txt_content and y_pos < 0:
                                text_obj.set_fontsize(17)
                            elif 'base value' in txt_content.lower() or 'higher' in txt_content.lower() or 'lower' in txt_content.lower() or 'f(x)' in txt_content.lower() or '\\leftarrow' in txt_content or '\\rightarrow' in txt_content:
                                text_obj.set_fontsize(18)
                                if y_pos > 0.9:
                                    text_obj.set_position((x_pos, y_pos + 0.03))
                            else:
                                text_obj.set_fontsize(16)

                fig_force.tight_layout(pad=0.8)
                st.pyplot(fig_force)
                plt.close(fig_force)

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
