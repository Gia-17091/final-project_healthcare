"""
TabPFN basic + PCA256 holdout 예측 결과 평가 스크립트.

원본 노트북(tabpfn_basic_pca500_holdout.ipynb)과 동일한 방식으로 y_true 를 복원해
  - Mean column-wise log-loss (대회 공식 지표, 원본 코드와 동일)
  - Per-target log-loss / AUC / AP
  - per_target.csv 와 조인된 진단 테이블
을 계산하고 결과를 CSV 로 저장한다.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# ---------- 설정 (원본 노트북과 동일) ----------
ROOT        = Path(__file__).parent
DATA_DIR    = ROOT
OUTPUT_DIR  = ROOT / "moa_outputs_tabpfn_basic_pca256_holdout"
PRED_PATH   = OUTPUT_DIR / "val_pred_tabpfn_basic_pca256.csv"
PERTGT_PATH = OUTPUT_DIR / "tabpfn_basic_pca256_per_target.csv"
EPS         = 1e-15
SEED        = 42
N_SPLITS    = 5
VAL_FOLD    = 0


def reconstruct_val_labels():
    """원본 holdout 노트북과 동일한 split 으로 val_split 의 y_true 를 복원."""
    train_features = pd.read_csv(DATA_DIR / "train_features.csv")
    train_targets  = pd.read_csv(DATA_DIR / "train_targets_scored.csv")
    target_cols    = [c for c in train_targets.columns if c != "sig_id"]

    df_full = train_features.merge(train_targets, on="sig_id")
    df_full = df_full[df_full["cp_type"] != "ctl_vehicle"].reset_index(drop=True)

    mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(mskf.split(df_full, df_full[target_cols]))
    _, va_idx = splits[VAL_FOLD]
    val_split = df_full.loc[va_idx].reset_index(drop=True)
    return val_split[["sig_id"] + target_cols], target_cols


def main():
    # 1) 예측 / 정답 로드
    print(f"[load] predictions: {PRED_PATH.name}")
    pred_df = pd.read_csv(PRED_PATH)

    print(f"[load] reconstructing y_true via MSKF(seed={SEED}, fold={VAL_FOLD})")
    truth_df, target_cols = reconstruct_val_labels()

    # sig_id 기준 정렬 후 확인
    merged = truth_df.merge(
        pred_df, on="sig_id", suffixes=("_true", "_pred"), how="inner"
    )
    assert len(merged) == len(truth_df) == len(pred_df), (
        f"sig_id mismatch: truth={len(truth_df)} pred={len(pred_df)} merged={len(merged)}"
    )
    print(f"[ok] matched {len(merged)} rows on sig_id")

    # float64 로 계산 — float32 에서는 1 - 1e-15 가 1.0 으로 반올림돼 log(0)=NaN 발생
    y_true = merged[[f"{c}_true" for c in target_cols]].values.astype(np.float64)
    y_pred = merged[[f"{c}_pred" for c in target_cols]].values.astype(np.float64)
    y_pred_clip = np.clip(y_pred, EPS, 1 - EPS)

    # 2) 공식 지표 — 원본 코드와 동일 (elementwise BCE → mean)
    official_loss = -(
        y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip)
    ).mean()
    print(f"\n=== Mean column-wise log-loss (원본 공식 지표) : {official_loss:.5f} ===")

    # 3) Per-target 지표
    per_tgt = []
    for j, col in enumerate(target_cols):
        yt = y_true[:, j]
        yp = y_pred_clip[:, j]
        n_pos = int(yt.sum())

        # log-loss (샘플이 한 클래스만 있어도 labels=[0,1] 로 고정 계산)
        ll = log_loss(yt, yp, labels=[0, 1])

        # AUC / AP 는 양성/음성이 모두 있어야 정의됨
        if 0 < n_pos < len(yt):
            auc = roc_auc_score(yt, yp)
            ap  = average_precision_score(yt, yp)
        else:
            auc = np.nan
            ap  = np.nan

        per_tgt.append({
            "target":        col,
            "n_pos_val":     n_pos,
            "pred_mean_val": float(yp.mean()),
            "logloss":       ll,
            "auc":           auc,
            "ap":            ap,
        })
    per_tgt_df = pd.DataFrame(per_tgt)

    # 4) 학습측 per_target 메타 와 조인 (n_pos=학습셋 양성 수, fit_predict_sec)
    if PERTGT_PATH.exists():
        meta = pd.read_csv(PERTGT_PATH).rename(
            columns={"n_pos": "n_pos_train", "pred_mean": "pred_mean_train"}
        )
        per_tgt_df = per_tgt_df.merge(meta, on="target", how="left")

    # 5) 요약 출력
    print("\n--- macro summary ---")
    print(f"  macro logloss : {per_tgt_df['logloss'].mean():.5f}")
    print(f"  macro AUC     : {per_tgt_df['auc'].mean(skipna=True):.5f} "
          f"(evaluated on {per_tgt_df['auc'].notna().sum()} / {len(per_tgt_df)} targets)")
    print(f"  macro AP      : {per_tgt_df['ap'].mean(skipna=True):.5f}")

    print("\n--- worst 10 targets by log-loss ---")
    print(per_tgt_df.sort_values("logloss", ascending=False)
                    .head(10)
                    .to_string(index=False))

    print("\n--- best 10 targets by AUC ---")
    print(per_tgt_df.dropna(subset=["auc"])
                    .sort_values("auc", ascending=False)
                    .head(10)
                    .to_string(index=False))

    # 6) 결과 저장
    out_path = OUTPUT_DIR / "eval_tabpfn_basic_pca256.csv"
    per_tgt_df.to_csv(out_path, index=False)

    summary_path = OUTPUT_DIR / "eval_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"official_mean_columnwise_logloss\t{official_loss:.6f}\n")
        f.write(f"macro_logloss\t{per_tgt_df['logloss'].mean():.6f}\n")
        f.write(f"macro_auc\t{per_tgt_df['auc'].mean(skipna=True):.6f}\n")
        f.write(f"macro_ap\t{per_tgt_df['ap'].mean(skipna=True):.6f}\n")
        f.write(f"n_val_rows\t{len(merged)}\n")
        f.write(f"n_targets\t{len(target_cols)}\n")

    print(f"\n[save] per-target eval  -> {out_path}")
    print(f"[save] summary          -> {summary_path}")


if __name__ == "__main__":
    main()
