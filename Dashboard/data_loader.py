# 1. 라이브러리 및 경로 설정
import streamlit as st
import pandas as pd
from pathlib import Path
from helpers import assign_reliability_grade

BASE = Path(__file__).parent.parent   # Dashboard/ 의 부모 = 1차] MoA 데이터/
DATA = BASE / "data"
MODEL_DIR = BASE / "Model"

# 2. 데이터 로딩 함수
 # 캐싱된 데이터 로딩 함수: 학습 데이터, 매핑 파일, 신뢰도 테이블
 # @st.cache_data 데코레이터로 캐싱하여 성능 최적화 수행.
@st.cache_data(show_spinner="학습 데이터 로딩 중...")
def load_train_data():
    try:
        # 파일이 실제로 존재하는지 확인하고 읽어옵니다.
        feat = pd.read_csv(DATA / "train_features.csv")
        tgt  = pd.read_csv(DATA / "train_targets_scored.csv")
        drug = pd.read_csv(DATA / "train_drug.csv")
        return feat, tgt, drug
    except FileNotFoundError as e:
        st.error(f"데이터 파일을 찾을 수 없습니다: {e}")
        st.stop()
        
@st.cache_data(show_spinner="매핑 파일 로딩 중...")
def load_mappings():
    gene = pd.read_csv(DATA / "gene_name.csv")
    cell = pd.read_csv(DATA / "cell_info.csv")
    return gene, cell


@st.cache_data(show_spinner="신뢰도 테이블 로딩 중...")
def load_reliability_table():
    df = pd.read_csv(MODEL_DIR / "eval_tabpfn_basic_pca256.csv")
    df["reliability"] = df.apply(
        lambda r: assign_reliability_grade(r["auc"], r["ap"], r["n_pos_val"]), axis=1
    )
    return df
