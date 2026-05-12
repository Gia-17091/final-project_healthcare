# 사이브 바 & 탭 조립 — app.py
import streamlit as st
from data_loader import load_train_data
import tab1_input
import tab2_moa
import tab3_toxicity
import tab4_ontarget
import tab5_rdview

# 1.페이지 설정(Page Config)
st.set_page_config(
    page_title="신규 화합물 MoA 및 세포독성 우선순위 대시보드",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. 사이드바(Sidebar)
with st.sidebar:
    st.header("분석 설정")
    compound_id = st.text_input("Compound ID", value="CMPD-2025-0512",
                                help="분석할 신규 화합물 식별자")
    cp_time = st.selectbox("cp_time (처리 시간)", [24, 48, 72], index=1,
                           help="약물 처리 후 측정 시간 (h)")
    cp_dose = st.selectbox("cp_dose (처리 농도)", ["D1", "D2"], index=1,
                           help="약물 처리 농도 조건")
    input_mode = st.radio("입력 방식", ["g만 입력", "g+c 입력"],
                          help="c 값 없으면 g→c 모델로 세포독성 자동 추정")
    st.divider()
    uploaded_file = st.file_uploader(
        "CSV 파일 업로드", type=["csv"],
        help="필수: cp_time, cp_dose, g-0~g-771 | 선택: compound_id, c-0~c-99",
    )
    use_sample = st.button("샘플 데이터 사용", use_container_width=True)
    st.caption("TIP: 샘플 3개 이상 포함 시 안정적 분석 가능합니다.")

# 3. 세션 상태(Session State)
 # 세션 상태 초기화: df_input (데이터프레임), analysis_ready (분석 준비 여부)
 # analysis_ready가 없으면, 새로 만들어서 False로 초기화
if "df_input" not in st.session_state:
    st.session_state.df_input       = None
if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False
 
 # 샘플 데이터 로드 및 세션 상태 업데이트
if use_sample:
    feat, _, _ = load_train_data()
    sample = feat[
        (feat["cp_type"] == "trt_cp") &
        (feat["cp_time"] == cp_time) &
        (feat["cp_dose"] == cp_dose)
    ].head(3).reset_index(drop=True)
    if len(sample) == 0:
        sample = feat[feat["cp_type"] == "trt_cp"].head(3).reset_index(drop=True)
    sample.insert(0, "compound_id", [f"{compound_id}-{i+1}" for i in range(len(sample))])
    st.session_state.df_input       = sample
    st.session_state.analysis_ready = True

 # 직접 파일 로드 시
if uploaded_file is not None:
    st.session_state.df_input       = __import__("pandas").read_csv(uploaded_file)
    st.session_state.analysis_ready = False

# 4. 헤더(Header)
st.title("신규 화합물 MoA 및 세포독성 우선순위 대시보드")
st.caption("Smiles 김하은, 손삼주 | 최종 프로젝트) 2024.05.14")

# 5. 탭(Tabs)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. 신규 화합물 입력",
    "2. MoA 후보 우선순위",
    "3. 세포독성 리스크",
    "4. On/Off-Target 예측",
    "5. 생물학적 및 연구개발(R&D) 관점",
])

with tab1:
    tab1_input.render(st.session_state.df_input, input_mode)

with tab2:
    tab2_moa.render(st.session_state.df_input)

with tab3:
    tab3_toxicity.render(st.session_state.df_input, input_mode)

with tab4:
    tab4_ontarget.render(st.session_state.df_input)

with tab5:
    tab5_rdview.render(st.session_state.df_input)
