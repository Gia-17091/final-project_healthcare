#  Tab 3: 세포 독성 리스크 (g→c Prediction & Flag Analysis)
 # - g→c 모델로 c 값 예측 (또는 c 직접 입력)

# 1. 라이브러리 설치
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from models import predict_c_from_g
from data_loader import load_train_data, load_mappings
from helpers import get_toxicity_risk

# 2. 독성 예측 기준 설정 (모델링 설정 기준과 동일)
THRESHOLD = -2.0

# 3. 데이터 검증 및 준비
def render(df: pd.DataFrame, input_mode: str) -> None:
    # 데이터가 없을 시, 안내 멘트 표시 후 종료
    if not st.session_state.analysis_ready or df is None:
        st.info("Tab 1에서 데이터를 먼저 업로드하세요.")
        return
    
    # 메타 데이터 및 세포주 매핑 정보 로드
    feat_ref, _, _ = load_train_data()
    c_cols_ref = [c for c in feat_ref.columns if c.startswith("c-")]
    _, cell_map_df = load_mappings()
    cell_dict = cell_map_df.set_index("rid")["ccle_name"].to_dict()

    # 독성 수치 예측 여부 결정: c 직접 입력 vs g→c 모델 예측
    use_actual_c = (input_mode == "g+c 입력" and all(c in df.columns for c in c_cols_ref))

    with st.spinner("세포독성 예측 중..."):
        if use_actual_c:
            pred_c_arr   = df[c_cols_ref].values.astype(float)
            source_label = "실측값 (c-feature 직접 입력)"
        else:
            pred_c_arr   = predict_c_from_g(df)
            source_label = "예측값 (g→c 모델)"

    mean_c     = pred_c_arr.mean(axis=0)
    flag_count = int((mean_c <= THRESHOLD).sum())
    flag_ratio = flag_count / len(mean_c) * 100
    min_c_val  = float(mean_c.min())
    risk_level, _, risk_desc = get_toxicity_risk(flag_count)

    # 1. 지표 카드
    st.subheader("독성 리스크 요약")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("독성 flag 세포주 수", f"{flag_count}개")
    c2.metric("최저 예측 c 값",     f"{min_c_val:.2f}")
    c3.metric("위험도 등급",        risk_level)
    c4.metric("독성 위험 비율",     f"{flag_ratio:.1f}%")

    st.divider() # 구분선
    
    # 2. 2단 레이아웃 설정(st.columns)
     # 왼쪽: 히스토그램, 오른쪽: Top 10 테이블
    col_l, col_r = st.columns(2) 

    # 시각화: 히스토그램
    with col_l:
        st.subheader("예측 c 값 분포")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=mean_c, nbinsx=40, name="c 값 분포",
            marker_color="steelblue", opacity=0.75,
        ))
        fig_hist.add_vline(
            x=THRESHOLD, line_dash="dash", line_color="red",
            annotation_text=f"세포독성 기준 (c <= {THRESHOLD})",
            annotation_position="top right",
        )
        fig_hist.update_layout(
            xaxis_title="예측 c 값", yaxis_title="세포주 수",
            height=340, margin=dict(t=30, b=30),
            showlegend=False
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # 시각화: Top 10 독성 세포주 테이블 
    with col_r:
        st.subheader("독성 위험 세포주 Top 10")
        top10_idx  = np.argsort(mean_c)[:10]
        top10_rows = []
        for rank, idx in enumerate(top10_idx, 1):
            c_name  = f"c-{idx}"
            ccle    = cell_dict.get(c_name, "Unknown")
            parts   = ccle.split("_")
            cell_ln = parts[0] if len(parts) > 1 else ccle
            tissue  = parts[-1] if len(parts) > 1 else "?"
            val     = float(mean_c[idx])
            top10_rows.append({
                "순위":        rank,
                "세포주 코드": c_name,
                "세포주명":    cell_ln,
                "조직":        tissue,
                "예측 c 값":   round(val, 3),
                "독성 flag":   "Yes" if val <= THRESHOLD else "No",
            })
        st.dataframe(pd.DataFrame(top10_rows), use_container_width=True, hide_index=True)

    # 시각화: 세포주별 위험 히트맵 (Top 20독성)
    st.subheader("세포주별 위험 Heatmap (Top 20 독성)")
    top20_idx    = np.argsort(mean_c)[:20]
    top20_labels = [
        cell_dict.get(f"c-{idx}", f"c-{idx}").split("_")[0]
        for idx in top20_idx
    ]

    # 히트맵 데이터 준비
    display_c = mean_c[top20_idx].reshape(1, -1)  # 1행 N열 형태로 변환
    fig_hm = go.Figure(data=go.Heatmap(
        z=display_c,
        x=top20_labels, y=["예측 c 값"],
        text=display_c,
        texttemplate="<b>%{text:.2f}</b>",
        textfont={"size": 10},
        colorscale="RdYlGn",
        zmid=-1.0, zmin=-3.5, zmax=0.5,
        colorbar=dict(title="c 값", thickness=15),
    ))

    fig_hm.update_layout(height=180, margin=dict(l=20, r=20, t=20, b=80))
    fig_hm.update_xaxes(tickangle=45)
    st.plotly_chart(fig_hm, use_container_width=True)

    # 4. 해석 및 권장 사항
    st.subheader("독성 해석 및 권장 사항")
    col_a, col_b = st.columns(2)
    with col_a:
        risk_df = pd.DataFrame({
            "독성 flag 세포주 수": ["0개", "1~5개", "6개 이상"],
            "위험도":             ["Low", "Medium", "High"],
            "해석":               ["뚜렷한 신호 없음", "주의 필요한 수준", "광범위한 독성 가능성"],
        })
        st.write("**위험도 분류 기준**")
        st.dataframe(risk_df, use_container_width=True, hide_index=True)
    with col_b:
        if risk_level == "High":
            st.error(f"High Risk — {risk_desc}\n\nMoA 후보 검증과 함께 독성 배제 실험(counter-screening) 병행 권장")
        elif risk_level == "Medium":
            st.warning(f"Medium Risk — {risk_desc}\n\n농도별 반응 실험(dose-response) 및 세포 생존율 assay 권장")
        else:
            st.success(f"Low Risk — {risk_desc} 상대적으로 안전한 프로파일로 보이나, 추가 검증 실험으로 확신 필요")

    st.caption(
        "c <= -2.0 기준은 프로젝트에서 정의한 in vitro 세포 반응 기반 warning signal입니다. "
        "임상 독성이나 장기 독성을 직접 의미하지 않습니다."
    )
