"""Tab 2 — MoA 후보 우선순위"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from models import predict_moa_proba, train_moa_model
from data_loader import load_reliability_table
from helpers import assign_decision_level


def render(df: pd.DataFrame) -> None:
    if not st.session_state.analysis_ready or df is None:
        st.info("Tab 1에서 데이터를 먼저 업로드하세요.")
        return

    with st.spinner("MoA 예측 중..."):
        proba_matrix, err = predict_moa_proba(df)

    if err:
        st.error(err)
        return

    _, moa_labels, _ = train_moa_model()
    mean_proba = proba_matrix.mean(axis=0)

    top_idx   = np.argsort(mean_proba)[::-1][:10]
    top_moas  = [moa_labels[i] for i in top_idx]
    top_probs = [float(mean_proba[i]) for i in top_idx]
    margin    = top_probs[0] - top_probs[1] if len(top_probs) > 1 else top_probs[0]

    rel_table = load_reliability_table()
    rel_dict  = rel_table.set_index("target")[
        ["auc", "ap", "n_pos_val", "reliability"]
    ].to_dict("index")

    high_cnt   = sum(1 for m in top_moas[:5]
                     if rel_dict.get(m, {}).get("reliability") == "High")
    verify_cnt = sum(1 for m, p in zip(top_moas[:5], top_probs[:5])
                     if rel_dict.get(m, {}).get("reliability") in ["High", "Medium"]
                     and p > 0.08)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top 1 MoA",        top_moas[0].replace("_", " "), f"확률 {top_probs[0]:.3f}")
    c2.metric("Top 3 평균 확률",   f"{np.mean(top_probs[:3]):.3f}")
    c3.metric("고신뢰(High) 후보", f"{high_cnt} / 5")
    c4.metric("검증 필요 후보",    f"{verify_cnt} / 5", f"margin {margin:.3f}")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Top 5 MoA 예측 확률")
        bar_df = pd.DataFrame({
            "MoA":         [m.replace("_", " ") for m in top_moas[:5]],
            "Probability": top_probs[:5],
        })
        fig_bar = px.bar(
            bar_df, x="Probability", y="MoA", orientation="h",
            color="Probability", color_continuous_scale="Blues",
            range_x=[0, max(1.0, top_probs[0] * 1.2)],
        )
        fig_bar.update_layout(
            height=300, yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        st.subheader("후보별 검증 우선순위")
        rows = []
        for rank, (moa, prob) in enumerate(zip(top_moas[:5], top_probs[:5]), 1):
            rel = rel_dict.get(moa, {})
            rows.append({
                "Rank":          rank,
                "Predicted MoA": moa.replace("_", " "),
                "Probability":   round(prob, 4),
                "Reliability":   rel.get("reliability", "Low"),
                "Val AUC":       round(rel.get("auc", 0), 3),
                "Val AP":        round(rel.get("ap", 0), 3),
                "Pos Count":     int(rel.get("n_pos_val", 0)),
                "Decision":      assign_decision_level(prob, rel.get("reliability", "Low")),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Top 10 MoA 상세 결과")
    full_rows = []
    for rank, (moa, prob) in enumerate(zip(top_moas, top_probs), 1):
        rel = rel_dict.get(moa, {})
        full_rows.append({
            "Rank":        rank,
            "MoA":         moa,
            "Probability": round(prob, 4),
            "Reliability": rel.get("reliability", "Low"),
            "AUC":         round(rel.get("auc", 0), 3),
            "AP":          round(rel.get("ap", 0), 3),
            "N_pos":       int(rel.get("n_pos_val", 0)),
            "우선순위":    assign_decision_level(prob, rel.get("reliability", "Low")),
        })
    st.dataframe(pd.DataFrame(full_rows), use_container_width=True, hide_index=True)

    with st.expander("신규 샘플에서 신뢰도를 어떻게 판단하나요?"):
        st.markdown("""
        1. **신규 샘플에는 정답이 없어** 개별 AUC/AP 계산 불가
        2. 대신 **과거 validation에서 계산한 MoA별 성능**으로 기본 신뢰도 산정
        3. 현재 샘플에서는 **예측 확률, rank, top1-top2 margin**으로 신호 강도 판단

        > **최종 우선순위 = 모델의 과거 신뢰도 × 현재 샘플의 상대적 신호 강도**
        """)
    st.warning("이 결과는 '이 MoA가 정답일 확률'이 아니라 **후속 실험 우선순위**입니다.")
