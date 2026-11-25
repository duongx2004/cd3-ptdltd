# 02_truc_quan.py
# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_DIR, OUT_DIR, N_MIN, RATING_MIN, RATING_MAX
from utils import save_fig


def run():
    df = pd.read_csv(os.path.join(DATA_DIR, "clean.csv"),
                     parse_dates=["time", "month"])

    brand_col = origin_col = None
    meta_path = os.path.join(OUT_DIR, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        brand_col = meta.get("BRAND_COL")
        origin_col = meta.get("ORIGIN_COL")

    # Phân phối điểm
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df["rating"].astype(int), color="#378ADD")
    plt.title("Phân phối điểm đánh giá")
    plt.xlabel("Điểm"); plt.ylabel("Số đánh giá")
    save_fig("phan_phoi_diem")

    # Số đánh giá và điểm TB theo tháng
    mon = df.groupby("month").agg(so_danh_gia=("rating", "size"),
                                   diem_tb=("rating", "mean"))
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(mon.index, mon["so_danh_gia"], color="#185FA5", marker="o", ms=3)
    ax1.set_ylabel("Số đánh giá", color="#185FA5")
    ax2 = ax1.twinx(); ax2.grid(False)
    ax2.plot(mon.index, mon["diem_tb"], color="#D85A30", marker="s", ms=3)
    ax2.set_ylabel("Điểm trung bình", color="#D85A30")
    plt.title("Số đánh giá và điểm trung bình theo tháng")
    save_fig("danh_gia_theo_thang")

    # Số đánh giá trên mỗi sản phẩm
    plt.figure(figsize=(6, 4))
    sns.histplot(df.groupby("item").size(), bins=40, log_scale=(True, False),
                 color="#1D9E75")
    plt.title("Số đánh giá trên mỗi sản phẩm (thang log)")
    plt.xlabel("Số đánh giá của sản phẩm"); plt.ylabel("Số sản phẩm")
    save_fig("danh_gia_moi_san_pham")

    def top_group_plot(col, label, fname):
        if not col or col not in df.columns:
            print(f"Bỏ qua: không có cột {label}.")
            return
        g = df.groupby(col).agg(so_danh_gia=("rating", "size"),
                                 diem_tb=("rating", "mean"),
                                 so_san_pham=("item", "nunique"))
        g = g[(g["so_danh_gia"] >= N_MIN) & (g.index != "Không rõ")] \
            .sort_values("so_danh_gia", ascending=False)
        if g.empty:
            print(f"Bỏ qua: không nhóm {label} nào đủ {N_MIN} đánh giá.")
            return
        top = g.head(10).iloc[::-1]
        fig, axes = plt.subplots(1, 2,
                                  figsize=(10, max(3, 0.45 * len(top) + 1.5)),
                                  sharey=True)
        axes[0].barh(top.index, top["so_danh_gia"], color="#378ADD")
        axes[0].set_title("Số đánh giá")
        axes[1].scatter(top["diem_tb"], top.index, color="#D85A30", s=60, zorder=3)
        for v, lab in zip(top["diem_tb"], top.index):
            axes[1].annotate(f"{v:.2f}", (v, lab), textcoords="offset points",
                             xytext=(0, 8), ha="center", fontsize=8)
        axes[1].set_title("Điểm trung bình")
        axes[1].set_xlim(max(RATING_MIN, top["diem_tb"].min() - 0.3),
                         min(RATING_MAX, top["diem_tb"].max() + 0.3))
        fig.suptitle(f"Top 10 {label} (đủ {N_MIN} đánh giá)")
        save_fig(fname)
        g.round(3).to_csv(
            os.path.join(OUT_DIR, f"thong_ke_theo_{label.replace(' ', '_')}.csv"),
            encoding="utf-8-sig")

    top_group_plot(brand_col,  "thương hiệu", "top_thuong_hieu")
    top_group_plot(origin_col, "xuất xứ",     "top_xuat_xu")

    # Số đánh giá trên mỗi người dùng
    pu = df.groupby("user").size().clip(upper=10)
    plt.figure(figsize=(6, 4))
    sns.countplot(x=pu, color="#7F77DD")
    plt.title("Số đánh giá trên mỗi người dùng (10 = từ 10 trở lên)")
    plt.xlabel("Số đánh giá"); plt.ylabel("Số người dùng")
    save_fig("danh_gia_moi_nguoi_dung")

    # Độ dài đánh giá theo điểm
    if df["text_len"].notna().any():
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df["rating"].astype(int), y=df["text_len"],
                    color="#9FE1CB", showfliers=False)
        plt.title("Độ dài đánh giá theo điểm")
        plt.xlabel("Điểm"); plt.ylabel("Độ dài (ký tự)")
        save_fig("do_dai_theo_diem")


if __name__ == "__main__":
    run()