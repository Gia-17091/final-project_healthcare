# 1. 라이브러리 및 경로 설정
import streamlit as st
import pandas as pd
from pathlib import Path
from helpers import assign_reliability_grade

current_file_path = Path(__file__).resolve()  # Dashboard/ 의 부모 = 1차] MoA 데이터/
BASE = current_file_path.parent.parent   
DATA = BASE / "data"
MODEL_DIR = BASE / "model"

# 2. 데이터 로딩 함수
 # 캐싱된 데이터 로딩 함수: 학습 데이터, 매핑 파일, 신뢰도 테이블
 # @st.cache_data 데코레이터로 캐싱하여 성능 최적화 수행.
@st.cache_data(show_spinner="학습 데이터 로딩 중...")
def load_train_data():
    # 초기값 설정 (정의되지 않음 에러 방지)
    feat, tgt, drug = None, None, None
    
    # train_features.csv 로드
    feat_path = DATA / "train_features.csv"
    if feat_path.exists():
        feat = pd.read_csv(feat_path)
    else:
        st.error(f"파일을 찾을 수 없습니다: {feat_path}")
    # tgt, drug 정의.    
    tgt_path = DATA / "train_targets_scored.csv"
    if tgt_path.exists():
        tgt = pd.read_csv(tgt_path)
   
    drug_path = DATA / "train_drug.csv"
    if drug_path.exists():
        drug = pd.read_csv(drug_path)

    feat = feat if feat is not None else pd.DataFrame()
    tgt = tgt if tgt is not None else pd.DataFrame()
    drug = drug if drug is not None else pd.DataFrame()

    return feat, tgt, drug

@st.cache_data(show_spinner="매핑 파일 로딩 중...")
def load_mappings():
    gene_path = DATA / "g_to_symbol_final.csv"
    cell_path = DATA / "c_to_cell_final.csv"
    
    gene = pd.read_csv(gene_path) if gene_path.exists() else pd.DataFrame()
    cell = pd.read_csv(cell_path) if cell_path.exists() else pd.DataFrame()
    return gene, cell


@st.cache_data(show_spinner="신뢰도 테이블 로딩 중...")
def load_reliability_table():
    df = pd.read_csv(MODEL_DIR / "eval_tabpfn_basic_pca256.csv")
    df["reliability"] = df.apply(
        lambda r: assign_reliability_grade(r["auc"], r["ap"], r["n_pos_val"]), axis=1
    )
    return df
