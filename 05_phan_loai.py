# 05_phan_loai.py
# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from config import (DATA_DIR, OUT_DIR, FIG_DIR, N_MIN_RISK, TRAIN_FRAC,
                    RANDOM_STATE, ITEM_CAT_COLS, BRAND_COL, SUMMARY)
from utils import save_fig, best_threshold, compute_metrics


def make_pipe(kind, cat_feats, num_feats):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
        ("num", StandardScaler(), num_feats)])
    if kind == "lr":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        clf = RandomForestClassifier(n_estimators=150, max_depth=12,
                                     min_samples_leaf=20, class_weight="balanced",
                                     n_jobs=-1, random_state=RANDOM_STATE)
    return Pipeline([("pre", pre), ("clf", clf)])


def _draw_table_image(df, path, title):
    """Vẽ DataFrame thành ảnh bảng đẹp (header xanh, hàng sọc xen kẽ)."""
    n_rows = len(df) + 1
    n_cols = len(df.columns)
    fig_w = max(9, n_cols * 1.6)
    fig_h = max(3.5, n_rows * 0.45 + 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # Header
    for j in range(n_cols):
        cell = table[(0, j)]
        cell.set_facecolor("#185FA5")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")

    # Body — zebra striping
    for i in range(1, n_rows):
        for j in range(n_cols):
            cell = table[(i, j)]
            cell.set_facecolor("#F0F4FA" if i % 2 == 0 else "white")
            cell.set_edgecolor("#C0C0C0")

    plt.title(title, fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.show()
    plt.close()


def run():
    df = pd.read_csv(os.path.join(DATA_DIR, "clean.csv"),
                     parse_dates=["time", "month"])
    d = df.sort_values("time", kind="mergesort").reset_index(drop=True)

    item_cat_cols, brand_col, name_col = [], None, None
    meta_path = os.path.join(OUT_DIR, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        item_cat_cols = meta.get("ITEM_CAT_COLS") or []
        brand_col = meta.get("BRAND_COL")
        name_col = meta.get("NAME_COL")
    if not item_cat_cols:
        item_cat_cols = ITEM_CAT_COLS or []
    if not brand_col:
        brand_col = BRAND_COL

    # Đặc trưng lịch sử
    g = d.groupby("item")
    d["prior_cnt"] = g.cumcount()
    d["prior_neg"] = g["neg"].cumsum() - d["neg"]
    d["prior_rate"] = d["prior_neg"] / d["prior_cnt"].replace(0, np.nan)
    d["log_prior_cnt"] = np.log1p(d["prior_cnt"])
    d["month_cat"] = d["month_num"].astype(str)

    cut = int(len(d) * TRAIN_FRAC)
    train_df, test_df = d.iloc[:cut].copy(), d.iloc[cut:].copy()
    print(f"Mốc chia: train {len(train_df):,} dòng (→ {train_df['time'].max()}) | "
          f"test {len(test_df):,} dòng (từ {test_df['time'].min()})")
    print(f"Tỷ lệ tiêu cực: train {train_df['neg'].mean()*100:.2f}% | "
          f"test {test_df['neg'].mean()*100:.2f}%")

    cats = [c for c in item_cat_cols if c in d.columns]
    if not cats:
        print("⚠️ Không có cột thuộc tính nào để dùng làm đặc trưng.")
    for c in cats:
        d[c] = d[c].astype(str)
        top = d.iloc[:cut][c].value_counts().head(30).index
        d[c] = d[c].where(d[c].isin(top), "Khác")
    train_df, test_df = d.iloc[:cut].copy(), d.iloc[cut:].copy()

    global_rate = float(train_df["neg"].mean())
    for part in (train_df, test_df):
        part["prior_rate"] = part["prior_rate"].fillna(global_rate)

    cat_feats = cats + ["month_cat"]
    num_feats = ["log_prior_cnt", "prior_rate"]
    feats = cat_feats + num_feats
    print("Đặc trưng:", feats)

    y_te = test_df["neg"].to_numpy()
    inner_cut = int(len(train_df) * 0.75)
    inner_tr, inner_va = train_df.iloc[:inner_cut], train_df.iloc[inner_cut:]

    results, scores, thresholds, fitted = [], {}, {}, {}

    results.append({"Mô hình": "Baseline: luôn đoán không tiêu cực",
                    **compute_metrics(y_te, np.zeros(len(y_te)), 0.5)})

    thr_b = best_threshold(inner_va["neg"].to_numpy(),
                            inner_va["prior_rate"].to_numpy())
    scores["Baseline: tỷ lệ tiêu cực lịch sử"] = test_df["prior_rate"].to_numpy()
    thresholds["Baseline: tỷ lệ tiêu cực lịch sử"] = thr_b
    results.append({"Mô hình": "Baseline: tỷ lệ tiêu cực lịch sử",
                    **compute_metrics(
                        y_te, scores["Baseline: tỷ lệ tiêu cực lịch sử"], thr_b)})

    for name, kind in [("Hồi quy logistic", "lr"),
                       ("Rừng ngẫu nhiên", "rf")]:
        p_in = make_pipe(kind, cat_feats, num_feats).fit(
            inner_tr[feats], inner_tr["neg"])
        thr = best_threshold(
            inner_va["neg"].to_numpy(),
            p_in.predict_proba(inner_va[feats])[:, 1])
        final = make_pipe(kind, cat_feats, num_feats).fit(
            train_df[feats], train_df["neg"])
        s = final.predict_proba(test_df[feats])[:, 1]
        fitted[name], scores[name], thresholds[name] = final, s, thr
        results.append({"Mô hình": name, **compute_metrics(y_te, s, thr)})
        print(f"Đã huấn luyện {name} (ngưỡng = {thr:.3f})")

    clf_table = pd.DataFrame(results).round(3)
    print("\nBẢNG. So sánh mô hình phân loại:\n",
          clf_table.to_string(index=False))
    clf_table.to_csv(os.path.join(OUT_DIR, "so_sanh_phan_loai.csv"),
                     index=False, encoding="utf-8-sig")

    best_clf = (clf_table[clf_table["Mô hình"]
                          .isin(["Hồi quy logistic", "Rừng ngẫu nhiên"])]
                .sort_values("ROC-AUC", ascending=False).iloc[0]["Mô hình"])
    print("Mô hình tốt nhất theo ROC-AUC:", best_clf)

    # Ma trận nhầm lẫn
    cm = confusion_matrix(y_te,
                          (scores[best_clf] >= thresholds[best_clf]).astype(int))
    plt.figure(figsize=(4.6, 4))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False,
                xticklabels=["Dự đoán: không", "Dự đoán: tiêu cực"],
                yticklabels=["Thực tế: không", "Thực tế: tiêu cực"])
    plt.title(f"Ma trận nhầm lẫn ({best_clf})")
    save_fig("ma_tran_nham_lan")

    # Đặc trưng quan trọng / hệ số
    pipe = fitted[best_clf]
    names = pipe.named_steps["pre"].get_feature_names_out()
    if best_clf == "Rừng ngẫu nhiên":
        imp = pd.Series(pipe.named_steps["clf"].feature_importances_,
                        index=names).sort_values(ascending=False).head(15)
        title = "Độ quan trọng đặc trưng (rừng ngẫu nhiên)"
    else:
        coef = pd.Series(pipe.named_steps["clf"].coef_[0], index=names)
        imp = pd.concat([coef.sort_values().head(8),
                         coef.sort_values().tail(8)])
        title = "Hệ số hồi quy logistic (dương: tăng nguy cơ)"
    plt.figure(figsize=(7, 5))
    imp.sort_values().plot(kind="barh", color="#7F77DD")
    plt.title(title)
    save_fig("dac_trung_quan_trong")
    imp.to_csv(os.path.join(OUT_DIR, "dac_trung_quan_trong.csv"),
               header=["gia_tri"], encoding="utf-8-sig")

    # Top sản phẩm nguy cơ cao
    tt = test_df.copy()
    tt["xac_suat_tieu_cuc"] = scores[best_clf]
    risk = tt.groupby("item").agg(
        so_danh_gia_kiem_tra=("neg", "size"),
        xac_suat_tieu_cuc_tb=("xac_suat_tieu_cuc", "mean"),
        ty_le_tieu_cuc_thuc_te=("neg", "mean"),
        diem_tb=("rating", "mean"))
    if name_col and name_col in tt.columns:
        risk = risk.join(tt.drop_duplicates("item")
                         .set_index("item")[name_col]
                         .rename("ten_san_pham"))
    if brand_col and brand_col in tt.columns:
        risk = risk.join(tt.drop_duplicates("item")
                         .set_index("item")[brand_col]
                         .rename("thuong_hieu"))
    risk = (risk[risk["so_danh_gia_kiem_tra"] >= N_MIN_RISK]
            .sort_values("xac_suat_tieu_cuc_tb", ascending=False))
    top_risk = risk.head(10).round(3)
    print(f"\nBẢNG. Top 10 sản phẩm nguy cơ cao "
          f"(đủ {N_MIN_RISK} đánh giá kiểm tra):\n", top_risk)
    top_risk.to_csv(os.path.join(OUT_DIR, "san_pham_nguy_co_cao.csv"),
                    encoding="utf-8-sig")
    risk.round(3).to_csv(os.path.join(OUT_DIR, "nguy_co_tat_ca_san_pham.csv"),
                         encoding="utf-8-sig")

    # ========== VẼ ẢNH BẢNG TOP 10 NG UY CƠ ==========
    if not top_risk.empty:
        # Chuẩn bị DataFrame hiển thị
        tbl = top_risk.copy().reset_index()
        if "item" in tbl.columns:
            tbl = tbl.rename(columns={"item": "Mã SP"})
        tbl.insert(0, "STT", range(1, len(tbl) + 1))

        cols_show = ["STT", "Mã SP"]
        rename_map = {}
        if "ten_san_pham" in tbl.columns:
            cols_show.append("ten_san_pham")
            rename_map["ten_san_pham"] = "Tên sản phẩm"
        if "thuong_hieu" in tbl.columns:
            cols_show.append("thuong_hieu")
            rename_map["thuong_hieu"] = "Thương hiệu"
        cols_show += ["so_danh_gia_kiem_tra", "xac_suat_tieu_cuc_tb",
                       "ty_le_tieu_cuc_thuc_te", "diem_tb"]
        rename_map.update({
            "so_danh_gia_kiem_tra": "Số ĐG (test)",
            "xac_suat_tieu_cuc_tb": "XS tiêu cực (dự đoán)",
            "ty_le_tieu_cuc_thuc_te": "Tỷ lệ tiêu cực (thực tế)",
            "diem_tb": "Điểm TB",
        })
        tbl = tbl[cols_show].rename(columns=rename_map)

        # Cắt ngắn tên SP cho vừa ô
        if "Tên sản phẩm" in tbl.columns:
            tbl["Tên sản phẩm"] = tbl["Tên sản phẩm"].astype(str).apply(
                lambda s: s[:25] + "..." if len(s) > 28 else s)

        # Format giá trị số
        for col in tbl.columns:
            if "XS tiêu cực" in col:
                tbl[col] = tbl[col].apply(lambda v: f"{v:.2f}")
            elif "Tỷ lệ tiêu cực" in col:
                tbl[col] = tbl[col].apply(lambda v: f"{v * 100:.1f}%")
            elif col == "Điểm TB":
                tbl[col] = tbl[col].apply(lambda v: f"{v:.2f}")

        _draw_table_image(
            tbl,
            os.path.join(FIG_DIR, "bang_top_10_nguy_co.png"),
            f"Top 10 sản phẩm có nguy cơ đánh giá tiêu cực cao nhất\n"
            f"(đủ {N_MIN_RISK} đánh giá trong tập kiểm tra)"
        )
    # ==================================================

    SUMMARY["phan_loai"] = dict(
        nguong_tieu_cuc=2,
        ty_le_tieu_cuc_train_pct=round(global_rate * 100, 2),
        ty_le_tieu_cuc_test_pct=round(float(y_te.mean() * 100), 2),
        mo_hinh_tot_nhat=best_clf,
        bang=clf_table.to_dict(orient="records"))
    return clf_table, top_risk


if __name__ == "__main__":
    run()