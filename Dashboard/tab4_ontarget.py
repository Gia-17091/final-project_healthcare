# Tab 4 : 폐 특화 On-Target / Off-Target 예측

# 1. 라이브러리 설치
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from models import predict_moa_proba, predict_c_from_g, train_moa_model
from data_loader import load_train_data, load_mappings

# 2. 폐암 관련 MoA 목록
LUNG_ON_TARGET_MOAS = [
    "egfr_inhibitor", "alk_inhibitor", "pi3k_inhibitor", "mtor_inhibitor",
    "kras_inhibitor", "met_inhibitor", "hdac_inhibitor", "cdk_inhibitor",
    "vegfr_inhibitor", "braf_inhibitor", "akt_inhibitor",
]
LUNG_OFF_TARGET_MOAS = [
    "proteasome_inhibitor", "nfkb_inhibitor", "tubulin_inhibitor",
    "dna_damage", "dna_alkylating_agent", "topoisomerase_inhibitor",
]

# 3. 폐 관련 세포주 패널
LUNG_CELL_PANEL = [
    ("A549", "폐 선암"),
    ("NCI-H1975", "폐 선암"),
    ("SK-MES-1", "폐 편평상피암"),
    ("BEAS-2B", "정상 폐 상피세포"),
    ("SAEC", "정상 폐 상피세포"),
    ("LUNG_CELL_LINE_X", "폐 조직 특이 세포주"),
]

# 4. 바이오마커 유전자 그룹 (지표명, 유전자 목록, 단위 종류)
BIOMARKERS = [
    ("세포 생존율 변화",              ["BCL2", "BAX",    "CASP3"], "pct"),
    ("염증 반응 지표 (IL-6 / TNF-α)", ["IL6",  "TNF",    "NFKB1"], "fold"),
    ("섬유화 관련 지표 (Col1A1)",     ["COL1A1","TGFB1", "FN1"],   "fold"),
    ("산화 스트레스 지표 (ROS)",      ["NFE2L2","HMOX1", "SOD2"],  "fold"),
    ("상피 손상 지표 (EpCAM)",        ["EPCAM", "CDH1",  "KRT18"], "fold"),
]

# 5. 폐 독성 위험 등급 기준
LUNG_RISK_THRESHOLDS = {
    "Low":    {"c_mean": -1.0},
    "Medium": {"c_mean": -2.0},
    "High":   {"c_mean": float("-inf")},
}

# 6. 대조군 평균 계산 (속도 최적화)
@st.cache_data(show_spinner=False)
def get_control_mean():
    feat_ref, _, _ = load_train_data()
    g_cols = [c for c in feat_ref.columns if c.startswith("g-")]
    # trt_cp(약물 처리군)의 평균 유전자 발현량 계산 후 Series로 반환
    control_mean = feat_ref[feat_ref["cp_type"] == "ctl"][g_cols].mean()
    return control_mean

# 7. render 함수 정의
def render(df: pd.DataFrame) -> None:
    # 데이터 미준비 시 안내 
    if not st.session_state.analysis_ready or df is None:
        st.info("Tab 1에서 데이터를 먼저 업로드하세요.")
        return

    # 사이드 바 설정
    st.sidebar.markdown("### 분석 파라미터 조절")
    tox_threshold = st.sidebar.slider("독성 판단 기준 (c-value)", -3.5, -1.0, -2.0, 0.1, key="tab4_tox_slider")
    fc_threshold = st.sidebar.slider("바이오마커 유의성 기준 (x)", 1.1, 2.5, 1.3, 0.1, key="tab4_fc_slider")
    st.subheader("폐 특화 On-Target / Off-Target 예측")

    #1) 화합물 개별 선택   
    if "drug_id" in df.columns:
        cmpd_list = df["drug_id"].tolist()
        id_type = "Drug ID"
    elif "compound_id" in df.columns:
        cmpd_list = df["compound_id"].tolist()
        id_type = "Compound ID"
    elif "sig_id" in df.columns:
        cmpd_list = df["sig_id"].tolist()
        id_type = "Signature ID"
    else:
        # 모든 ID 컬럼이 없을 경우에만 Sample 번호를 부여합니다.
        cmpd_list = [f"Sample {i+1}" for i in range(len(df))]
        id_type = "화합물"

    selected_cmpd = st.selectbox("화합물 선택", options=cmpd_list)
    selected_idx = cmpd_list.index(selected_cmpd)

    with st.spinner("폐 특화 On/Off-Target 분석 중..."):
        proba_matrix, err = predict_moa_proba(df)
        pred_c_arr = predict_c_from_g(df)
        gene_map_df, cell_map_df = load_mappings()
        gene_dict = gene_map_df.set_index("kaggle_col")["symbol"].to_dict()
        cell_dict = cell_map_df.set_index("rid")["ccle_name"].to_dict()
        _, moa_labels, _ = train_moa_model()

    # 2) 선택한 화합물만 추출
    target_proba = proba_matrix[selected_idx]
    target_c     = pred_c_arr[selected_idx]

    # 3) On/Off-Target 점수 계산
    on_probs  = [float(target_proba[moa_labels.index(m)]) for m in LUNG_ON_TARGET_MOAS  if m in moa_labels]
    off_probs = [float(target_proba[moa_labels.index(m)]) for m in LUNG_OFF_TARGET_MOAS if m in moa_labels]

    on_raw  = float(np.mean(on_probs))  if on_probs  else 0.01
    off_raw = float(np.mean(off_probs)) if off_probs else 0.01
    on_score  = int(round(on_raw / (on_raw + off_raw) * 100))
    off_score = 100 - on_score

    # 4) 폐 세포주 독성 평균
    lung_keywords = ["A549", "NCI-H", "SK-MES", "BEAS", "SAEC", "LUNG"]
    lung_idx = [i for i in range(100) if any(kw in cell_dict.get(f"c-{i}", "").upper() for kw in lung_keywords)]
    lung_c_mean = float(target_c[lung_idx].mean()) if lung_idx else float(target_c.mean())

    if lung_c_mean <= tox_threshold:
        lung_risk, lung_risk_desc = "High", f"{tox_threshold} 이하의 심각한 폐 조직 손상"
    elif lung_c_mean <= (tox_threshold + 1.0):
        lung_risk, lung_risk_desc = "Medium", "주의가 필요한 폐 조직 손상 신호"
    else:
        lung_risk, lung_risk_desc = "Low", "낮은 폐 조직 독성 리스크"

    if on_score >= 65:
        judgment, judgment_sub = "On-Target", "정상 약효 우세"
    elif on_score >= 45:
        judgment, judgment_sub = "중립",       "약효·독성 균형"
    else:
        judgment, judgment_sub = "Off-Target", "비특이적 반응 우세"

    # 대시보드 확인
    # 1] 지표 카드
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("On-Target 점수",     f"{on_score}%",  "0 ~ 100 범위")
    m2.metric("Off-Target 점수",    f"{off_score}%", "0 ~ 100 범위")
    m3.metric("On/Off 판정", "On-Target" if on_score >= 65 else "중립" if on_score >= 45 else "Off-Target")
    m4.metric("폐 독성 위험 등급",  lung_risk,       lung_risk_desc)

    st.divider()

    # 2] 상단: 가우시안 분포 / 핵심 지표 요약 
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("On/Off-Target 예측 분포")
        # 분포계산
        x = np.linspace(0, 100, 400)
        gauss_on  = np.exp(-0.5 * ((x - 72) / 14) ** 2)
        gauss_off = np.exp(-0.5 * ((x - 28) / 14) ** 2)

        # 그래프 시각화
        fig_gauss = go.Figure()

        # Off-Target 영역 (빨간색)
        fig_gauss.add_trace(go.Scatter(x=x, y=gauss_off, fill="tozeroy",
                                       name="Off-Target 영역(부작용 위험)", 
                                       line=dict(color="red", width=2),
                                       fillcolor="rgba(220,50,50,0.20)"))
        
        # On-Target 영역 (초록색)
        fig_gauss.add_trace(go.Scatter(x=x, y=gauss_on, fill="tozeroy", 
                                       name="On-Target 영역(정상 약효)", 
                                       line=dict(color="green", width=2),
                                       fillcolor="rgba(50,180,50,0.20)"))
        
        # 현재 점수 표시(파란색)
        fig_gauss.add_vline(x=on_score, line_dash="dash", 
                            line_color="#1f77b4", line_width=2.5,
                            annotation=dict(text=f"{on_score}", 
                                            font=dict(size=15, color="#1f77b4"),
                                            bgcolor="white", bordercolor="#1f77b4"))
        
        # 레이아웃 조정
        fig_gauss.update_layout(
            xaxis_title="예측 점수 (0 ~ 100)",
            yaxis=dict(showticklabels=False),
            height=320, margin=dict(t=10, b=40, l=10, r=10),
            legend=dict(orientation="h", y=-0.28)
        )

        # 화면에 보이도록 출력
        st.plotly_chart(fig_gauss, use_container_width=True)

    # 핵심 지표 요약 테이블 (폐 특화 바이오마커)
    with col_r:
        st.subheader("폐 특화 핵심 지표 요약")

        # 유전자 심볼 → feature 인덱스 역방향 매핑
        g_cols_in_df = [c for c in df.columns if c.startswith("g-")]
        symbol_to_idx = {sym: g_cols_in_df.index(col) for col, sym in gene_dict.items() if col in g_cols_in_df}
        
        # 데이터 준비 및 비교 연산
        trt_mean_series = get_control_mean() 
        trt_mean_arr = trt_mean_series[g_cols_in_df].values
        
        # 선택한 약물의 유전자 발현량
        target_g = df.iloc[selected_idx][g_cols_in_df].values.astype(float).flatten()
        g_diff = target_g - trt_mean_arr

        # 바이오 마커 그룹별 변화량 계산
        def _bm_diff(symbols):
            vals = [
                g_diff[symbol_to_idx[s]]
                for s in symbols
                if s in symbol_to_idx and symbol_to_idx[s] < len(g_diff)
            ]
            return float(np.mean(vals)) if vals else 0.0
        
        # 테이블 행 생성 및 해석 로직
        bm_rows = []
        for name, syms, kind in BIOMARKERS:
            d = _bm_diff(syms)

            if kind == "pct":
                val_str = f"{d * 10:.0f}%"
                interp  = "정상 범위 내 세포 사멸 → 약효 기대" if d < 0 else "세포 증식 증가 가능성"
            else:
                fold = 1.0 + d * 0.1
                val_str = f"{fold:.1f}x"    # x는 몇 배를 의미. 
                
                # 경고 메시지
                if "염증" in name:
                    interp = "낮은 염증 반응 → 부작용 가능성 낮음" if fold <= 1.3 else "염증 반응 증가 주의"
                elif "섬유화" in name:
                    interp = "섬유화 위험 낮음" if fold <= 1.2 else "섬유화 위험 경고"
                elif "산화" in name:
                    interp = "산화스트레스 변화 미미" if abs(d) < 0.5 else "산화 스트레스 증가"
                else:
                    interp = "상피 손상 징후 미미" if fold >= 0.85 else "상피 손상 경고"
            bm_rows.append({"지표 항목": name, "값 (변화)": val_str, "해석": interp})

        st.dataframe(pd.DataFrame(bm_rows), use_container_width=True, hide_index=True, height=215)

    st.divider()

    # 3] 중단: 히트맵 / On·Off 근거
    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.subheader("세포주별 폐 독성 위험 히트맵 (예측 점수)")
        hm_x, hm_z = [], []
        for cname, tissue in LUNG_CELL_PANEL:
            matched = next(
                (i for i in range(100) if cname.upper() in cell_dict.get(f"c-{i}", "").upper()),
                None,
            )
            c_val      = float(target_c[matched]) if matched is not None else lung_c_mean
            risk_score = max(0, min(100, int(-c_val * 33)))
            hm_x.append(f"{cname}\n({tissue})")
            hm_z.append(risk_score)

        fig_hm = go.Figure(go.Heatmap(
            z=[hm_z], x=hm_x, y=["예측 점수 (0~100)"],
            colorscale="RdYlGn_r", zmin=0, zmax=100,
            text=[[str(v) for v in hm_z]],
            texttemplate="<b>%{text}</b>",
            colorbar=dict(title="위험", thickness=14, len=0.85),
        ))
        fig_hm.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=110))
        fig_hm.update_xaxes(tickangle=-45, tickfont=dict(size=9))
        st.plotly_chart(fig_hm, use_container_width=True)

    with col_b2:
        st.subheader("예측 기전(MoA) 근거")
        oc1, oc2 = st.columns(2)
        
        on_evidence = sorted([{"근거": m.replace("_inhibitor", "").upper(), "확률": round(float(target_proba[moa_labels.index(m)]), 2)} for m in LUNG_ON_TARGET_MOAS if m in moa_labels], key=lambda x: x["확률"], reverse=True)
        off_evidence = sorted([{"근거": m.replace("_inhibitor", "").upper(), "확률": round(float(target_proba[moa_labels.index(m)]), 2)} for m in LUNG_OFF_TARGET_MOAS if m in moa_labels], key=lambda x: x["확률"], reverse=True)
        
        with oc1:
            st.markdown("**On-Target (폐암 표적)**")
            st.dataframe(pd.DataFrame(on_evidence[:5]), use_container_width=True, hide_index=True)
        with oc2:
            st.markdown("**Off-Target (부작용 리스크)**")
            st.dataframe(pd.DataFrame(off_evidence[:5]), use_container_width=True, hide_index=True)

    st.divider()

    # 4] 하단: 종합 해석 / 권장 사항
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.subheader("종합 해석")
        if on_score >= 65:
            st.success(f"{selected_cmpd}는 폐 특화 모델 기준 On-Target 가능성이 높습니다.")
        elif on_score >= 45:
            st.warning(f"{selected_cmpd}는 On-Target과 Off-Target이 균형을 이루고 있습니다.")
        else:
            st.error(f"{selected_cmpd}는 Off-Target 반응이 우세할 가능성이 있습니다.")

        top_on_label = on_evidence[0]["근거"] if on_evidence else "-"
        st.write(
            f"{selected_cmpd}는 {top_on_label} 효과가 뚜렷하며, "
            f"정상 폐 세포에서는 독성 신호가 {'낮고' if lung_risk != 'High' else '높고'} "
            f"염증 및 섬유화 지표 변화가 {'미미합니다' if lung_risk == 'Low' else '관찰됩니다'}. "
            f"따라서 본 약물은 {'정상 약효 우세한 On-Target' if on_score >= 65 else 'Off-Target'} "
            "작용이 주로 작용할 것으로 예측됩니다."
        )

        fig_donut = go.Figure(go.Pie(
            values=[on_score, off_score],
            labels=["On-Target", "Off-Target"],
            hole=0.65,
            marker_colors=["#2ecc71", "#e74c3c"],
            textinfo="label+percent",
        ))
        fig_donut.update_layout(
            height=260, margin=dict(t=10, b=10, l=10, r=10),
            annotations=[dict(
                text=f"<b>{on_score}%</b><br>On-Target",
                x=0.5, y=0.5, showarrow=False, font=dict(size=14),
            )],
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_c2:
        st.subheader("권장 사항")
        flag_count = int((target_c <= -2.0).sum())
        recs = [
            ("현재 농도 수준에서 in-vivo 폐 독성 위험 낮음",      lung_risk == "Low"),
            ("고농도 / 장기 노출 조건에서 추가 안전성 검토 권장",  True),
            ("산화 스트레스 마커 모니터링 권장 (ROS, NRF2)",       flag_count >= 3),
            ("면역세포 반응 및 섬유화 관련 장기 추적 관찰 권장",   lung_risk in ["Medium", "High"]),
        ]
        for rec_text, active in recs:
            icon = "✅" if active else "☐"
            st.write(f"{icon} {rec_text}")

        st.caption(
            "위 권장 사항은 예측 결과 기반 가설 생성용 제안입니다. "
            "반드시 실험적 검증을 통해 확인해야 합니다."
        )
