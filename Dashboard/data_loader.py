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
    file_name = "train_features.csv"
    full_path = DATA / file_name
 
    if not full_path.exists():
        st.error(f"파일을 찾을 수 없습니다. 시도한 경로: {full_path}")
        raise FileNotFoundError(f"경로 확인: {full_path}")
     
    feat = pd.read_csv(full_path)
    # tgt, drug 정의.    
    try:
        tgt = pd.read_csv(DATA / "train_targets_scored.csv") 
        drug = pd.read_csv(DATA / "train_drug.csv")       
    except:
        tgt = None
        drug = None
    
    # 이제 feat, tgt, drug 세 개 모두 존재하므로 NameError가 나지 않습니다.
    return feat, tgt, drug


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
