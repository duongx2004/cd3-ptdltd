# utils.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from config import FIG_DIR

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110


def read_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".txt"):
        return pd.read_csv(path, encoding_errors="replace")
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t", encoding_errors="replace")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in (".json", ".jsonl"):
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    raise ValueError(f"Chưa hỗ trợ định dạng: {path}")


def parse_time(s):
    if pd.api.types.is_numeric_dtype(s):
        med = s.dropna().median()
        unit = "s" if med < 1e11 else "ms" if med < 1e14 else "us" if med < 1e17 else "ns"
        return pd.to_datetime(s, unit=unit, errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def clean_text(s):
    return (s.astype("string").str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.title()
            .fillna("Không rõ")
            .replace("", "Không rõ")
            .astype(str))


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{name}.png"), bbox_inches="tight")
    plt.show()
    plt.close()


def step(name, d, log_list):
    if len(d) == 0:
        raise RuntimeError(f"Bước '{name}' làm mất toàn bộ dữ liệu. "
                           f"Kiểm tra tên cột, thang điểm, định dạng thời gian.")
    log_list.append({"buoc": name, "so_dong_con_lai": len(d)})
    print(f"{name:<55s} -> {len(d):>9,} dòng")
    return d


def best_threshold(y, s):
    if int(np.sum(y)) == 0:
        return 0.5
    cands = np.unique(np.quantile(s, np.linspace(0.30, 0.995, 80)))
    f1s = [f1_score(y, (s >= t).astype(int), zero_division=0) for t in cands]
    return float(cands[int(np.argmax(f1s))])


def compute_metrics(y, s, thr):
    pred = (s >= thr).astype(int)
    try:
        auc = roc_auc_score(y, s)
    except ValueError:
        auc = np.nan
    return dict(Precision=precision_score(y, pred, zero_division=0),
                Recall=recall_score(y, pred, zero_division=0),
                F1=f1_score(y, pred, zero_division=0),
                **{"ROC-AUC": auc})


# ---------- Các hàm dự báo ----------
def future_index(train, h):
    return pd.date_range(train.index[-1] + pd.DateOffset(months=1),
                         periods=h, freq="MS")


def f_naive(train, h):
    return np.repeat(float(train.iloc[-1]), h)


def f_snaive(train, h):
    if len(train) < 12:
        return np.full(h, np.nan)
    last = train.iloc[-12:].to_numpy(dtype=float)
    return np.array([last[i % 12] for i in range(h)])


def f_ma3(train, h):
    return np.repeat(float(train.iloc[-3:].mean()), h)


def f_linear(train, h):
    from sklearn.linear_model import LinearRegression
    season = len(train) >= 24

    def X(dates, t0):
        t = np.arange(t0, t0 + len(dates)).reshape(-1, 1)
        return np.hstack([t, np.eye(12)[dates.month - 1][:, 1:]]) if season else t

    m = LinearRegression().fit(X(train.index, 0), train.to_numpy(dtype=float))
    return np.clip(m.predict(X(future_index(train, h), len(train))), 0, None)