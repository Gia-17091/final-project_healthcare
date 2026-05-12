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
    # 디버깅: 파일이 존재하지 않을 시, 시도한 경로 화면에 출력
    if not full_path.exists():
        st.error(f"파일을 찾을 수 없습니다. 시도한 경로: {full_path}")
        # 혹시 몰라 상위 폴더의 파일 목록을 출력해봅니다.
        if DATA_PATH.exists():
            st.write(f"data 폴더 내 파일 목록: {list(DATA_PATH.glob('*'))}")
        else:
            st.write(f"data 폴더 자체가 존재하지 않습니다. 루트 목록: {list(BASE_DIR.glob('*'))}")
        raise FileNotFoundError(f"경로를 확인하세요: {full_path}")
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
