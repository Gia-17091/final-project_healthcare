# Tab 1 — 신규 화합물 데이터 업로드 및 검증

# 1. 라이브러리 설치
import streamlit as st
import pandas as pd

# 2. 데이터 업로드 및 검증 UI 구현
def render(df: pd.DataFrame | None, input_mode: str) -> None:
    st.subheader("신규 화합물 데이터 업로드 및 검증")

    if df is None:
        st.info("사이드바에서 CSV 파일을 업로드하거나 '샘플 데이터 사용' 버튼을 클릭하세요.")
        col_a, col_b = st.columns(2)
        with col_a:
            req_df = pd.DataFrame({
                "컬럼":      ["compound_id", "cp_time", "cp_dose", "g-0 ~ g-771", "c-0 ~ c-99"],
                "필수 여부": ["권장", "필수", "필수", "필수", "선택"],
                "설명": [
                    "신규 화합물 또는 샘플 식별자",
                    "약물 처리 후 측정 시간 (24/48/72)",
                    "약물 처리 농도 조건 (D1/D2)",
                    "유전자 발현 feature (772개)",
                    "세포 생존율 feature (없으면 g→c 모델로 추정)",
                ],
            })
            st.write("**필수 입력 컬럼**")
            st.dataframe(req_df, use_container_width=True, hide_index=True)
        with col_b:
            st.write("**분석 파이프라인**")
            st.info(
                "Step 1. 신규 화합물 입력\n\n"
                "Step 2. MoA 예측 모델 (g-feature → 206개 MoA 확률)\n\n"
                "Step 3. g→c 모델 (g-feature → c-feature 예측)\n\n"
                "Step 4. 세포독성 flag (c <= -2.0)\n\n"
                "Step 5. 후속 검증 우선순위 종합"
            )
        return

    G_COLS_ALL = [f"g-{i}" for i in range(772)]
    C_COLS_ALL = [f"c-{i}" for i in range(100)]

    n_samples   = len(df)
    n_g_found   = sum(1 for c in df.columns if c.startswith("g-"))
    has_g       = all(c in df.columns for c in G_COLS_ALL)
    has_c       = all(c in df.columns for c in C_COLS_ALL)
    has_cp_time = "cp_time" in df.columns
    has_cp_dose = "cp_dose" in df.columns
    valid_time  = has_cp_time and all(v in [24, 48, 72] for v in df["cp_time"].unique())
    valid_dose  = has_cp_dose and all(v in ["D1", "D2"]  for v in df["cp_dose"].unique())
    ready       = has_g and has_cp_time and has_cp_dose and valid_time and valid_dose

    if ready:
        st.session_state.analysis_ready = True

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("입력 샘플 수",   f"{n_samples}개")
    c2.metric("사용 feature",   f"{n_g_found}개", "g-0 ~ g-771")
    c3.metric("입력 방식",      input_mode)
    c4.metric("분석 준비 상태", "준비 완료" if ready else "확인 필요")

    st.write("**샘플 데이터 미리보기**")
    preview_cols = (["compound_id"] if "compound_id" in df.columns else []) + \
                   (["cp_time", "cp_dose"] if has_cp_time else []) + \
                   ["g-0", "g-1", "g-2"]
    if has_c:
        preview_cols += ["c-0", "c-1"]
    st.dataframe(df[[c for c in preview_cols if c in df.columns]].head(5),
                 use_container_width=True)

    st.write("**필수 컬럼 검증**")
    checks = [
        ("cp_time 컬럼 존재",               has_cp_time),
        ("cp_dose 컬럼 존재",               has_cp_dose),
        (f"g-feature 772개 ({n_g_found}개)", has_g),
        ("c-feature 100개 (선택)",           has_c),
        ("cp_time 값 유효 (24/48/72)",       valid_time),
        ("cp_dose 값 유효 (D1/D2)",          valid_dose),
    ]
    cl, cr = st.columns(2)
    for i, (label, ok) in enumerate(checks):
        icon = "✅OK" if ok else ("(선택)" if "선택" in label else "X")
        (cl if i % 2 == 0 else cr).write(f"[{icon}] {label}")
    
    st.info(
        "본 대시보드는 화합물의 유전자 발현(g)을 기반으로 가능한 MoA 후보와 "
        "세포독성 경고 신호를 예측·우선순위화합니다."
    )
    st.warning(
        "주의: 이 대시보드는 가설 생성형 도구입니다. "
        "예측 결과는 최종적인 생물학적 결론이나 의사결정 근거가 될 수 없으며, "
        "반드시 실험적 검증을 통해 확인해야 합니다."
    )
