# 1. 라이브러리 설정
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from data_loader import load_train_data

# 2. 모델 학습 함수 — @st.cache_resource 적용
 # 1) MoA 예측 모델: g-feature → 206개 MoA 확률
@st.cache_resource(show_spinner="MoA 예측 모델 학습 중...")
def train_moa_model():
    feat, tgt, _ = load_train_data()
    trt  = feat[feat["cp_type"] == "trt_cp"].set_index("sig_id")
    y_df = tgt.set_index("sig_id")
    common = trt.index.intersection(y_df.index)
    trt, y_df = trt.loc[common], y_df.loc[common]

    g_cols     = [c for c in trt.columns if c.startswith("g-")]
    moa_labels = y_df.columns.tolist()
    X = trt[g_cols].values.astype(np.float32)
    y = y_df.values.astype(np.int8)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=80, random_state=42)),
        ("clf",    MultiOutputClassifier(
            LogisticRegression(C=0.05, max_iter=200, solver="saga",
                               random_state=42, n_jobs=1),
            n_jobs=-1,
        )),
    ])
    pipe.fit(X, y)
    return pipe, moa_labels, g_cols

  # 2) 세포독성 예측 모델: g-feature → c-feature 예측값
@st.cache_resource(show_spinner="세포독성 예측 모델(g→c) 학습 중...")
def train_toxicity_model():
    feat, _, _ = load_train_data()
    trt    = feat[feat["cp_type"] == "trt_cp"].copy()
    g_cols = [c for c in trt.columns if c.startswith("g-")]
    c_cols = [c for c in trt.columns if c.startswith("c-")]
    trt    = trt.dropna(subset=c_cols)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=80, random_state=42)),
        ("reg",    Ridge(alpha=1.0)),
    ])
    pipe.fit(trt[g_cols].values.astype(np.float32),
             trt[c_cols].values.astype(np.float32))
    return pipe, g_cols, c_cols

# 3. 실전 예측: MoA 예측 모델(g-feature → 206개 MoA 확률 행렬 반환)
 # 오류 시 (None, 메시지) 반환
def predict_moa_proba(df_input):
    pipe, moa_labels, g_cols = train_moa_model()
    missing = [c for c in g_cols if c not in df_input.columns]
    if missing:
        return None, f"g-feature 누락: {missing[:5]} ..."
    X   = df_input[g_cols].values.astype(np.float32)
    X_s = pipe.named_steps["scaler"].transform(X)
    X_p = pipe.named_steps["pca"].transform(X_s)
    clf = pipe.named_steps["clf"]
    proba = np.column_stack([est.predict_proba(X_p)[:, 1] for est in clf.estimators_])
    return proba, None  # shape (n_samples, 206)

# 4. 실전 예측: 세포독성 예측 모델(g-feature → c-feature 예측값 반환)
def predict_c_from_g(df_input) -> np.ndarray:
    pipe, g_cols, _ = train_toxicity_model()
    X = df_input[[c for c in g_cols if c in df_input.columns]].values.astype(np.float32)
    return pipe.predict(X)
