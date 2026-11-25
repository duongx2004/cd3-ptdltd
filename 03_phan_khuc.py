# 03_phan_khuc.py
# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_DIR, OUT_DIR, N_MIN, SUMMARY
from utils import save_fig


def run():
    df = pd.read_csv(os.path.join(DATA_DIR, "clean.csv"),
                     parse_dates=["time", "month"])

    name_col = None
    meta_path = os.path.join(OUT_DIR, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        name_col = meta.get("NAME_COL")

    # (A) Đường cong tích lũy
    cnt = df.groupby("item").size().sort_values(ascending=False)
    cum_items = np.arange(1, len(cnt) + 1) / len(cnt)
    cum_reviews = cnt.cumsum().to_numpy() / cnt.sum()
    share = {f"top_{int(p*100)}pct_san_pham":
             round(float(cum_reviews[max(int(len(cnt) * p) - 1, 0)] * 100), 1)
             for p in (0.1, 0.2, 0.5)}
    print("Tỷ lệ đánh giá ở nhóm sản phẩm đầu bảng (%):", share)

    plt.figure(figsize=(6, 4))
    plt.plot(cum_items * 100, cum_reviews * 100, color="#185FA5")
    plt.plot([0, 100], [0, 100], "--", color="gray")
    plt.xlabel("% sản phẩm (xếp theo số đánh giá giảm dần)")
    plt.ylabel("% đánh giá tích lũy")
    plt.title("Mức độ tập trung của đánh giá")
    save_fig("tap_trung_pareto")

    # (B) Phân khúc sản phẩm
    stat = df.groupby("item").agg(so_danh_gia=("rating", "size"),
                                   diem_tb=("rating", "mean"))
    if name_col and name_col in df.columns:
        stat = stat.join(
            df.drop_duplicates("item").set_index("item")[name_col]
            .rename("ten_san_pham"))

    elig = stat[stat["so_danh_gia"] >= N_MIN].copy()
    if elig.empty:
        print(f"⚠️ Không sản phẩm nào đủ N_MIN={N_MIN} đánh giá. Bỏ qua.")
        SUMMARY["phan_khuc"] = dict(n_min=N_MIN, so_san_pham_du_dieu_kien=0)
        return None, None

    pop_thr = elig["so_danh_gia"].median()
    qual_thr = df["rating"].mean()
    print(f"Sản phẩm đủ n_min={N_MIN}: {len(elig)}/{len(stat)} | "
          f"ngưỡng phổ biến (trung vị) = {pop_thr:.0f} | ngưỡng điểm = {qual_thr:.2f}")

    hi_pop = elig["so_danh_gia"] >= pop_thr
    hi_q   = elig["diem_tb"] >= qual_thr
    elig["phan_khuc"] = np.select(
        [hi_pop & hi_q, hi_pop & ~hi_q, ~hi_pop & hi_q],
        ["Nổi bật", "Phổ biến cần cải thiện", "Tiềm năng"],
        default="Cần xem xét")

    seg = elig.groupby("phan_khuc").agg(
        so_san_pham=("so_danh_gia", "size"),
        so_danh_gia_tb=("so_danh_gia", "mean"),
        diem_tb=("diem_tb", "mean"),
        tong_danh_gia=("so_danh_gia", "sum"))
    seg["pct_san_pham"] = (seg["so_san_pham"] / seg["so_san_pham"].sum() * 100).round(1)
    seg["pct_tong_danh_gia"] = (seg["tong_danh_gia"] / seg["tong_danh_gia"].sum() * 100).round(1)
    seg = seg.round(2)
    print("\nBẢNG. Phân khúc sản phẩm:\n", seg)
    seg.to_csv(os.path.join(OUT_DIR, "phan_khuc_san_pham.csv"),
               encoding="utf-8-sig")
    elig.to_csv(os.path.join(OUT_DIR, "chi_tiet_phan_khuc_tung_san_pham.csv"),
                encoding="utf-8-sig")

    pal = {"Nổi bật": "#1D9E75", "Tiềm năng": "#378ADD",
           "Phổ biến cần cải thiện": "#D85A30", "Cần xem xét": "#888780"}
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=elig, x="so_danh_gia", y="diem_tb",
                    hue="phan_khuc", palette=pal, s=35)
    plt.xscale("log")
    plt.axvline(pop_thr, ls="--", c="gray")
    plt.axhline(qual_thr, ls="--", c="gray")
    plt.title("Phân khúc sản phẩm theo độ phổ biến và chất lượng")
    plt.xlabel("Số đánh giá (thang log)"); plt.ylabel("Điểm trung bình")
    plt.legend(title="", fontsize=8)
    save_fig("phan_khuc_san_pham")

    SUMMARY["phan_khuc"] = dict(
        n_min=N_MIN, so_san_pham_du_dieu_kien=int(len(elig)),
        nguong_pho_bien=float(pop_thr), nguong_diem=round(float(qual_thr), 3),
        tap_trung=share)
    return seg, elig


if __name__ == "__main__":
    run()