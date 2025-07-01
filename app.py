import streamlit as st
import joblib
import pandas as pd
import numpy as np



# 设置页面标题
st.set_page_config(
    page_title="Prediction of P growth rate by biochar in soils",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Prediction of P growth rate by biochar in soils")

# ==============================
# 自定义CSS样式（优化版）
# ==============================
st.markdown("""
<style>
/* ===== 全局样式 ===== */
* {
    font-family: 'Arial', 'Helvetica', sans-serif !important;
}

.stApp {
    background: linear-gradient(135deg, #f0f7ff 0%, #e1edff 100%);
    background-attachment: fixed;
}

/* ===== 标题样式 ===== */
h1, h2, h3 {
    color: #1E3F66 !important;
    font-weight: 700 !important;
}
h1 {
    border-bottom: 3px solid #2E5A8C;
    padding-bottom: 12px;
    margin-bottom: 30px;
}

/* ===== 分区标题 ===== */
.section-header {
    background: linear-gradient(to right, #1E3F66, #2E5A8C);
    color: white !important;
    padding: 12px 20px;
    border-radius: 8px;
    margin-top: 25px;
    margin-bottom: 15px;
    font-size: 1.2rem;
    font-weight: bold;
    box-shadow: 0 4px 6px rgba(30, 63, 102, 0.2);
}

/* ===== 输入框样式 ===== */
.stTextInput input, .stNumberInput input, .stSelectbox select {
    background-color: white !important;
    border: 2px solid #a8c6f0 !important;
    border-radius: 8px !important;
    padding: 10px 12px !important;
    font-size: 1rem !important;
    color: #1E3F66 !important;
    box-shadow: 0 2px 5px rgba(168, 198, 240, 0.3) !important;
}

.stTextInput label, .stNumberInput label, .stSelectbox label {
    font-weight: 600 !important;
    color: #1E3F66 !important;
    margin-bottom: 8px !important;
    font-size: 1.05rem !important;
}

input::placeholder {
    color: #7d9cc5 !important;
    opacity: 1 !important;
    font-style: italic;
}

.stFormSubmitButton > button {
    background: linear-gradient(to right, #AD0F0E, #AD0F0E) !important;
    color: white !important;
    font-size: 1.5rem !important;
    font-weight: bold !important;
    padding: 20px 40px !important;
    border-radius: 12px !important;
    width: 480px !important;
    height: 55px !important;
    margin: 0 auto !important;
    display: flex !important;              /* 关键：flex 布局 */
    align-items: center !important;        /* 垂直居中 */
    justify-content: center !important;    /* 水平居中 */
    text-align: center !important;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2) !important;
    transition: 0.2s ease-in-out all !important;
    line-height: 1.2 !important;           /* 避免字体被压扁 */
}

/* 🔧 放大按钮内部的文本元素 */
button[data-testid="stBaseButton-primaryFormSubmit"] * {
    font-size: 1.25rem !important;
    font-weight: bold !important;
    text-align: center !important;
}


.stFormSubmitButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(46, 90, 140, 0.4) !important;
}

/* ===== 结果卡片 ===== */
.result-section {
    background: linear-gradient(to bottom right, white, #f5f9ff);
    border-left: 5px solid #4A90E2;
    padding: 25px 30px;
    border-radius: 10px;
    box-shadow: 0 6px 15px rgba(74, 144, 226, 0.15);
    margin: 20px 0;
    border: 1px solid #d4e3fc;
}

.result-title {
    color: #1E3F66 !important;
    font-size: 1.5rem !important;
    margin-bottom: 20px !important;
    border-bottom: 2px solid #4A90E2;
    padding-bottom: 10px;
}

.result-value {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #1E3F66 !important;
    text-align: center;
    padding: 15px;
    background: rgba(74, 144, 226, 0.1);
    border-radius: 8px;
    margin: 10px 0;
}

.result-label {
    font-weight: 600 !important;
    color: #2E5A8C !important;
    font-size: 1.1rem !important;
    margin-bottom: 5px !important;
}

.error-value {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #d9534f !important; /* 红色系用于误差 */
    text-align: center;
    padding: 15px;
    background: rgba(217, 83, 79, 0.1);
    border-radius: 8px;
    margin: 10px 0;
}

/* ===== 页脚样式 ===== */
.footer {
    text-align: center;
    padding: 20px;
    margin-top: 40px;
    color: #5a7eb3;
    font-size: 0.9rem;
    border-top: 1px solid #d4e3fc;
}

/* ===== 预测区域布局 ===== */
.prediction-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 15px 0;
}

.prediction-item {
    flex: 1;
}

.prediction-label {
    font-size: 1.1rem;
    font-weight: 600;
    color: #2E5A8C;
    margin-bottom: 10px;
}

/* 真实值输入框样式 */
.actual-value-input {
    background-color: white !important;
    border: 2px solid #a8c6f0 !important;
    border-radius: 8px !important;
    padding: 12px 15px !important;
    font-size: 1.1rem !important;
    color: #1E3F66 !important;
    box-shadow: 0 2px 5px rgba(168, 198, 240, 0.3) !important;
    width: 100%;
    text-align: center;
}
            

# -------------------------------
/* Prediction 区域布局优化 */
.prediction-block {
    max-width: 720px;
    margin: 0 auto;
    padding: 10px 20px;
    background: #fdfdff;
    border: 1px solid #d4e3fc;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0, 80, 160, 0.06);
}

.prediction-block div[data-testid="stElementContainer"],
.prediction-block div[data-testid="stMarkdownContainer"],
.prediction-block .stFormSubmitButton,
.prediction-block .result-value {
    margin-top: 4px !important;
    margin-bottom: 4px !important;
    padding-top: 0px !important;
    padding-bottom: 0px !important;
}

.prediction-label {
    font-size: 1.1rem;
    font-weight: 600;
    color: #2E5A8C;
    margin-bottom: 10px;
}

.result-value {
    font-size: 1.6rem !important;
    font-weight: 700;
    color: #1E3F66;
    text-align: center;
    padding: 10px;
    background: rgba(74, 144, 226, 0.1);
    border-radius: 8px;
    border: 1px solid #d4e3fc;
}



# -------------------------------
</style>
""", unsafe_allow_html=True)

# 引入Font Awesome图标
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">', unsafe_allow_html=True)

# ==============================
# 模型加载函数
# ==============================
@st.cache_resource
def load_model_resources():
    try:
        model = joblib.load('./predict/Model/rf_model.pkl')
        scaler = joblib.load('./predict/Model/x_scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"加载资源失败: {str(e)}")
        st.stop()

model, scaler = load_model_resources()

# ==============================
# 输入表单设计
# ==============================
with st.form("prediction_form"):
    # 生物炭属性部分
    st.markdown('<div class="section-header">Biochar Property</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ash = st.number_input('Ash content (%)', value=10.5, min_value=0.0, max_value=100.0, step=0.1)
        cec = st.number_input('Cation exchange capacity (cmol(+)/kg)', value=18.6, min_value=0.0, step=0.1)
        nbc = st.number_input('Nitrogen content (%)', value=5.1, min_value=0.0, step=0.1)
        
    with col2:
        pbc = st.number_input('Phosphorus content (%)', value=6.3, min_value=0.0, step=0.1)
        opbc = st.number_input('Olsen-P content (mg/kg)', value=4.8, min_value=0.0, step=0.1)
        ssa = st.number_input('Specific surface area (m²/g)', value=112, min_value=0)
        
    with col3:
        biochar_ph = st.number_input('Biochar pH', value=6.8, min_value=0.0, max_value=14.0, step=0.1)
        carbon_content = st.number_input('Carbon content (%)', value=1.9, min_value=0.0, step=0.1)
        biochar_ec = st.number_input('Electrical conductivity (dS/m)', value=0.32, min_value=0.0, step=0.01)
    
    # 土壤属性部分
    st.markdown('<div class="section-header">Soil Property</div>', unsafe_allow_html=True)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        soil_ph = st.number_input('Soil pH', value=6.8, min_value=0.0, max_value=14.0, step=0.1)
        tp = st.number_input('Total phosphorus content (g/kg)', value=1.25, min_value=0.0, step=0.01)
        opsoil = st.number_input('Soil Olsen-P content (mg/kg)', value=0.53, min_value=0.0, step=0.01)
        
    with col5:
        tc = st.number_input('Total carbon content (%)', value=1.9, min_value=0.0, step=0.1)
        oc = st.number_input('Organic carbon content (g/kg)', value=0.87, min_value=0.0, step=0.01)
        soil_ec = st.number_input('Electrical conductivity (µS/cm)', value=320.0, min_value=0.0, step=1.0)
        
    with col6:
        soil_texture_options = {
            "Loam soil": 1,
            "Silt soil": 2,
            "Coarse-textured soil": 3,
            "Fine-textured soil": 4,
            "Medium-textured soil": 5,
            "Clay soil": 6
        }
        soil_texture = st.selectbox(
            'Soil texture',
            options=list(soil_texture_options.keys()),
            index=0
        )
    
    # 实验条件部分
    st.markdown('<div class="section-header">Experimental Condition</div>', unsafe_allow_html=True)
    
    col7, col8, col9 = st.columns(3)
    
    with col7:
        pyrolysis_temp = st.number_input('Biochar pyrolysis temperature (℃)', value=500, min_value=0)
        pyrolysis_time = st.number_input('Biochar pyrolysis time (h)', value=4, min_value=0)
        
    with col8:
        application_rate = st.number_input('Biochar application rate (%)', value=2, min_value=0)
        incubation_time = st.number_input('Incubation time (d)', value=30, min_value=1)
        
    with col9:
        fertilizer_options = {
            "No fertilization": 1,
            "Inorganic fertilization": 2,
            "Organic fertilization": 3,
            "Compound fertilization": 4
        }
        fertilizer_type = st.selectbox(
            'Fertilizer type',
            options=list(fertilizer_options.keys()),
            index=0
        )
    
    # =============================================
    # 预测区域设计（简洁版）
    # =============================================
    st.markdown('<div class="prediction-block">', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Prediction</div>', unsafe_allow_html=True)
    
    # 使用一行布局
    st.markdown('<div class="prediction-row">', unsafe_allow_html=True)
    
    # 真实值输入框（简洁设计）
    st.markdown('<div class="prediction-item">', unsafe_allow_html=True)
    st.markdown('<div class="prediction-label">Actual P growth rate (optional)</div>', unsafe_allow_html=True)
    actual_value = st.number_input(
        "", 
        value=0.0, 
        step=0.1,
        key="actual_value",
        label_visibility="collapsed",
        format="%.4f"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 预测按钮
    # 在按钮前后加上自定义居中容器
    st.markdown('<div style="display: flex; justify-content: center;">', unsafe_allow_html=True)
    # st.markdown('<div class="prediction-item stFormSubmitButton">', unsafe_allow_html=True)
    # st.markdown('<div class="prediction-label">&nbsp;</div>', unsafe_allow_html=True)  # 空白标签对齐
    submit_button = st.form_submit_button(
        "Start to predict P growth rate", 
        use_container_width=False,
        type="primary"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 预测结果展示（简洁设计）
    st.markdown('<div class="prediction-item">', unsafe_allow_html=True)
    st.markdown('<div class="prediction-label">Predicted P growth rate</div>', unsafe_allow_html=True)
    
    # 初始化预测值
    prediction_value = 0.0
    if submit_button:
        # 构建输入数据框
        input_data = pd.DataFrame([{
            'Incubation time': incubation_time,
            'T': pyrolysis_temp,
            'Time': pyrolysis_time,
            'Add': application_rate,
            'Ash': ash,
            'CBC': carbon_content,
            'CEC': cec,
            'NBC': nbc,
            'PBC': pbc,
            'OPBC': opbc,
            'SSA': ssa,
            'pHsoil': soil_ph,
            'TP': tp,
            'OPsoil': opsoil,
            'TC': tc,
            'OC': oc,
            'Fertilizer type': fertilizer_options[fertilizer_type],
            'Soil texture': soil_texture_options[soil_texture],
            'ECsoil': biochar_ec
        }])
        
        try:
            # 特征缩放
            input_scaled = pd.DataFrame(
                scaler.transform(input_data),
                columns=input_data.columns
            )
            
            # 预测
            prediction_value = model.predict(input_scaled)[0]
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            prediction_value = 0.0
    
    # 显示预测结果
    st.markdown(f'<div class="result-value" style="min-height: 50px; display: flex; align-items: center; justify-content: center;">{prediction_value:.4f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # 结束prediction-row
    st.markdown('</div>', unsafe_allow_html=True)  # prediction-wrapper 结束

    # =============================================
    # 误差分析区域（如果有真实值输入）
    # =============================================
    if submit_button and actual_value > 0:
        error = abs(prediction_value - actual_value)
        relative_error = (error / actual_value) * 100 if actual_value != 0 else 0
        
        st.markdown('<div class="section-header">Error Analysis</div>', unsafe_allow_html=True)
        
        # 使用两列布局
        col_error1, col_error2 = st.columns(2)
        
        with col_error1:
            st.markdown('<div class="result-label">Absolute Error</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="error-value">{error:.4f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-label">| Predicted - Actual |</div>', unsafe_allow_html=True)
        
        with col_error2:
            st.markdown('<div class="result-label">Relative Error</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="error-value">{relative_error:.2f}%</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-label">(Absolute Error / Actual Value) × 100%</div>', unsafe_allow_html=True)

# ==============================
# 页脚信息
# ==============================
st.markdown("""
<div class="footer">
   Growth Rate Prediction v1.0 | © 2025
</div>
""", unsafe_allow_html=True)

