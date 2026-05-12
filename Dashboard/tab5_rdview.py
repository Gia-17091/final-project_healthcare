# Tab 5: 생물학적 및 연구개발(R&D) 관점

# 1. 라이브러리 설치
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from models import predict_moa_proba, predict_c_from_g, train_moa_model
from data_loader import load_train_data, load_mappings, load_reliability_table
from helpers import get_moa_knowledge, get_toxicity_risk

# 2. render 함수 정의
def render(df: pd.DataFrame) -> None:
    # 데이터 미준비 시 안내 멘트
    if not st.session_state.analysis_ready or df is None:
        st.info("Tab 1에서 데이터를 먼저 업로드하세요.")
        return
    
    # 1) 참조 데이터 및 매핑 로드
    feat_ref, _, _ = load_train_data()
    g_cols_ref = [c for c in feat_ref.columns if c.startswith("g-")]
    gene_map_df, cell_map_df = load_mappings()
    gene_dict = gene_map_df.set_index("kaggle_col")["symbol"].to_dict()
    cell_dict = cell_map_df.set_index("rid")["ccle_name"].to_dict()

    # 2) MoA 예측 및 독성 flag 계산
    with st.spinner("생물학적 해석 데이터 생성 중..."):
        proba_matrix, err = predict_moa_proba(df)
        if err:
            st.error(err)
            return

    _, moa_labels, _ = train_moa_model()
    mean_proba = proba_matrix.mean(axis=0)
    top_idx    = np.argsort(mean_proba)[::-1][:5]
    top5_list  = [(moa_labels[i], float(mean_proba[i])) for i in top_idx]

    # 3) 세포독성 예측 및 위험도 판별
    pred_c_arr = predict_c_from_g(df)
    mean_c     = pred_c_arr.mean(axis=0)
    flag_count = int((mean_c <= -2.0).sum())
    risk_level, _, _ = get_toxicity_risk(flag_count)

    # 4) 유전자 발현량 차이 계산(g-)
    feat_trt_mean = feat_ref[feat_ref["cp_type"] == "trt_cp"][g_cols_ref].mean().values
    df_g_mean     = df[[c for c in g_cols_ref if c in df.columns]].mean().values
    g_diff        = df_g_mean - feat_trt_mean
    top_g_idx     = np.argsort(np.abs(g_diff))[::-1][:8]

    top_moa_name, top_moa_prob = top5_list[0]
    top_info = get_moa_knowledge(top_moa_name)

    # Streamlit UI 구성
    # 1. 지표 카드
    c1, c2, c3 = st.columns(3)
    c1.metric("Top 1 MoA",    top_moa_name.replace("_", " "), f"확률 {top_moa_prob:.3f}")
    c2.metric("주요 독성 신호", f"{flag_count}개",               "세포독성 flag 감지")
    c3.metric("R&D 추천 수",   "3가지",                         "후속 실험 제안")

    st.divider()
    p1, p2 = st.columns(2)

    # 2. MoA 생물학적 해석
    with p1:
        st.subheader("1. MoA 생물학적 해석")
        st.markdown(f"**Top 예측 MoA**: `{top_moa_name.replace('_', ' ')}`")
        st.write(f"**생물학적 의미**: {top_info['desc']}")
        st.write(f"**예상 신호**: {top_info['signal']}")
        st.write("**Top 3 해석 포인트**")
        for moa, prob in top5_list[:3]:
            info = get_moa_knowledge(moa)
            st.write(f"- **{moa.replace('_', ' ')}** (p={prob:.3f}) — {info['desc'][:55]}...")

    # 3. MoA 예측 + 독성 flag 연결 차트
    with p2:
        st.subheader("2. MoA 예측 + 세포독성 flag 연결")
        
        # 상위 10개 MoA 추출
        rel_idx    = load_reliability_table().set_index("target")["reliability"].to_dict()
        top10_i    = np.argsort(mean_proba)[::-1][:10]
        bar_moas   = [moa_labels[i] for i in top10_i]
        bar_probs  = [float(mean_proba[i]) for i in top10_i]
        color_map  = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#95a5a6"}
        bar_colors = [color_map.get(rel_idx.get(m, "Low"), "#95a5a6") for m in bar_moas]

        fig_b2 = go.Figure(go.Bar(
            x=[m.replace("_", " ")[:18] for m in bar_moas],
            y=bar_probs,
            marker_color=bar_colors,
            text=[f"{p:.3f}" for p in bar_probs],
            textposition="outside",
        ))
        fig_b2.update_layout(
            height=300, xaxis_tickangle=45,
            yaxis_title="Predicted Probability",
            title=f"독성 리스크: {risk_level} ({flag_count}개 flag)",
            showlegend=False, margin=dict(t=50, b=10),
        )
        st.plotly_chart(fig_b2, use_container_width=True)
    st.divider()
    p3, p4 = st.columns(2)

    # 4. g-feature 반응 패턴 해석
    with p3:
        st.subheader("3. g-feature 반응 패턴 해석")
        g_rows = []
        for idx in top_g_idx:
            g_name = f"g-{idx}"
            sym    = gene_dict.get(g_name, "Unknown")
            diff   = float(g_diff[idx])
            g_rows.append({
                "Gene":                  sym,
                "방향":                  "up" if diff > 0 else "down",
                "log2FC (vs train avg)": round(diff, 3),
            })
        st.dataframe(pd.DataFrame(g_rows), use_container_width=True, hide_index=True)
        st.caption("위 유전자 변화는 가설 생성용 feature이며, 인과관계를 입증하는 증거가 아닙니다.")

    # 4. 세포주별 독성 반응 해석
    with p4:
        st.subheader("4. 세포주별 독성 반응 해석")
        tox_rows = []
        for idx in np.argsort(mean_c)[:6]:
            c_name  = f"c-{idx}"
            ccle    = cell_dict.get(c_name, "Unknown")
            parts   = ccle.split("_")
            cell_ln = parts[0] if len(parts) > 1 else ccle
            tissue  = parts[-1] if len(parts) > 1 else "?"
            val     = float(mean_c[idx])
            risk    = "High" if val <= -2.0 else ("Watch" if val <= -1.0 else "Low")
            tox_rows.append({
                "Cell line":   cell_ln, "Tissue": tissue,
                "Predicted c": round(val, 3), "Risk": risk,
            })
        st.dataframe(pd.DataFrame(tox_rows), use_container_width=True, hide_index=True)
        st.caption("이 결과는 세포 수준의 예측 데이터이며, 장기 수준의 임상 독성을 의미하지 않습니다.")

    # 5. R&D 시사점 및 후속 실험 제안
    st.divider()
    st.subheader("5. R&D 시사점 및 후속 실험")
    recs = [f"1. {top_moa_name.replace('_', ' ').title()} 우선 검증: {top_info['assay']}"]

    if flag_count >= 6:
        top3_cells = [
            cell_dict.get(f"c-{idx}", f"c-{idx}").split("_")[0]
            for idx in np.argsort(mean_c)[:3]
        ]
        recs.append(
            f"2. 독성 세포주 심층 스크리닝: {', '.join(top3_cells)} 등 "
            "High 세포주에서 counter-screening 및 dose-response 확인"
        )
    elif flag_count >= 1:
        recs.append(
            "2. 독성 신호 추가 확인: 독성 flag 세포주에서 세포 생존율 assay "
            "(MTT/CCK-8) 및 dose-response curve 측정"
        )
    else:
        recs.append("2. 현재 조건에서 뚜렷한 세포독성 경고 신호 없음 — 추가 농도/시간 조건 확인 권장")

    if len(top5_list) >= 2:
        sec_moa, sec_prob = top5_list[1]
        recs.append(
            f"3. 대안 MoA 검증 고려: {sec_moa.replace('_', ' ')} (p={sec_prob:.3f}) "
            f"— {get_moa_knowledge(sec_moa)['assay'][:60]}..."
        )
    # 화면에 추천 리스트 출력
    for rec in recs:
        st.write(f"**{rec}**")

    st.warning(
        "해석 한계: 이 탭의 도메인 해석은 확정적 생물학 결론이 아니라 가설 생성형 설명입니다. "
        "유전자 변화와 MoA 사이의 인과관계를 입증하려면 별도 실험과 외부 생물학 자료 검증이 필요합니다."
    )

    # 6. 데이터 및 리포트 다운로드
    st.divider()
    st.subheader("6. 데이터 및 리포트 다운로드")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("**1. 분석 데이터 다운로드**")
        dl_df = df.copy()

        # 배열 연산 오류 방지(np.array변환 & axis 지정)
        moa_labels_arr = np.array(moa_labels)
        dl_df["predicted_moa"] = moa_labels_arr[np.argmax(proba_matrix, axis=1)]
        dl_df["predicted_moa_prob"] = proba_matrix.max(axis=1)
        
        # 한글 깨짐 방지
        csv_data = dl_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button( "주요 유전자 변화량 다운로드(csv)",
            data=csv_data,
            file_name="analysis_data.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption("업로드한 데이터에 예측된 MoA와 세포독성 점수를 추가한 csv 파일입니다.")
    
    # 마크다운 리포트 생성 및 다운로드 버튼 구현
    with col_d2:
        st.markdown("**2. 분석 리포트 다운로드**")
        report_md = f"""# MoA 및 독성 분석 리포트

## 1. 주요 결과 요약
- **Top 1 MoA**: {top_moa_name.replace('_', ' ')} (확률 {top_moa_prob:.3f})
- **세포독성 flag 수**: {flag_count}개 (위험도: {risk_level})

## 2. MoA 생물학적 해석
- Top 예측 MoA: {top_moa_name.replace('_', ' ')}
- 생물학적 작용 의미: {top_info['desc']}
- 예상 신호: {top_info['signal']}

## 3. R&D 시사점 및 후속 실험 제안
"""
        for rec in recs:
            report_md += f"- {rec}\n"

        report_md += """
---
## 4. 해석 한계
이 리포트의 도메인 해석은 확정적 생물학 결론이 아니라 가설 생성형 설명입니다. 유전자 변화와 MoA 사이의 인과관계를 입증하려면 별도 실험과 외부 생물학 자료 검증이 필요합니다.
"""
        st.download_button( "분석 리포트 다운로드(md)",
            data=report_md.encode("utf-8"),
            file_name="analysis_report.md",
            mime="text/markdown",
            use_container_width=True
        )
        st.caption("현재 분석 결과를 요약한 마크다운 형식의 리포트입니다.")