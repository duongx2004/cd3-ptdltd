# 04_du_bao.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import DATA_DIR, OUT_DIR, TEST_MONTHS, HORIZON, DROP_EDGE_MONTHS, SUMMARY
from utils import save_fig, future_index, f_naive, f_snaive, f_ma3, f_linear

MODELS = {"Naive": f_naive, "Naive theo mùa": f_snaive,
          "Trung bình trượt 3 tháng": f_ma3, "Hồi quy tuyến tính": f_linear}


def run():
    df = pd.read_csv(os.path.join(DATA_DIR, "clean.csv"),
                     parse_dates=["time", "month"])
    monthly = df.groupby("month").size().sort_index()
    monthly = monthly.reindex(
        pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS"),
        fill_value=0)
    if DROP_EDGE_MONTHS and len(monthly) > 2:
        monthly = monthly.iloc[1:-1]

    print(f"Chuỗi dự báo: {len(monthly)} tháng, "
          f"từ {monthly.index[0].date()} đến {monthly.index[-1].date()}")

    if len(monthly) < TEST_MONTHS + 6:
        print("⚠️ Chuỗi quá ngắn để dự báo có ý nghĩa — bỏ qua phần này.")
        return None, None

    train, test = monthly.iloc[:-TEST_MONTHS], monthly.iloc[-TEST_MONTHS:]
    if len(train) < 24:
        print("⚠️ Dưới 24 tháng huấn luyện: hồi quy chỉ có xu hướng.")

    rows, preds = [], {}
    for name, fn in MODELS.items():
        p = fn(train, len(test))
        preds[name] = p
        if np.isnan(p).any():
            rows.append({"Mô hình": name, "MAE": np.nan, "RMSE": np.nan,
                         "MAPE (%)": np.nan})
            continue
        y = test.to_numpy(dtype=float)
        mape = (np.mean(np.abs(y - p)[y != 0] / y[y != 0]) * 100
                if (y != 0).any() else np.nan)
        rows.append({"Mô hình": name,
                     "MAE": mean_absolute_error(y, p),
                     "RMSE": mean_squared_error(y, p) ** 0.5,
                     "MAPE (%)": mape})

    fc_table = pd.DataFrame(rows).round(2)
    print(f"\nBẢNG. So sánh mô hình (kiểm tra {TEST_MONTHS} tháng cuối):\n",
          fc_table.to_string(index=False))
    fc_table.to_csv(os.path.join(OUT_DIR, "so_sanh_du_bao.csv"),
                    index=False, encoding="utf-8-sig")

    valid = fc_table.dropna(subset=["MAE"])
    if valid.empty:
        print("⚠️ Không mô hình nào chạy được. Bỏ qua dự báo tương lai.")
        return fc_table, None

    best = valid.sort_values("MAE").iloc[0]["Mô hình"]
    print("Mô hình tốt nhất theo MAE:", best)
    fut = MODELS[best](monthly, HORIZON)
    fut_idx = future_index(monthly, HORIZON)
    rmse_best = float(fc_table.set_index("Mô hình").loc[best, "RMSE"])
    fc_future = pd.DataFrame({
        "thang": fut_idx.strftime("%Y-%m"),
        "du_bao": np.round(fut, 0),
        "can_duoi": np.clip(np.round(fut - rmse_best, 0), 0, None),
        "can_tren": np.round(fut + rmse_best, 0)})
    print("\nDự báo các tháng tới (± RMSE tham khảo):\n",
          fc_future.to_string(index=False))
    fc_future.to_csv(os.path.join(OUT_DIR, "du_bao_cac_thang_toi.csv"),
                     index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 4.5))
    plt.plot(monthly.index, monthly.values, color="#185FA5",
             marker="o", ms=3, label="Thực tế")
    plt.plot(test.index, preds[best], color="#D85A30",
             marker="s", ms=4, label=f"Dự báo kiểm tra ({best})")
    plt.plot(fut_idx, fut, color="#1D9E75",
             marker="^", ms=4, ls="--", label="Dự báo tương lai")
    plt.fill_between(fut_idx, fc_future["can_duoi"],
                     fc_future["can_tren"], color="#1D9E75", alpha=0.15)
    plt.axvline(test.index[0], color="gray", ls=":")
    plt.title("Số đánh giá theo tháng: thực tế và dự báo")
    plt.ylabel("Số đánh giá"); plt.legend()
    save_fig("du_bao_theo_thang")

    SUMMARY["du_bao"] = dict(
        so_thang=int(len(monthly)), thang_kiem_tra=TEST_MONTHS,
        mo_hinh_tot_nhat=best,
        bang=fc_table.to_dict(orient="records"),
        du_bao_tuong_lai=fc_future.to_dict(orient="records"))
    return fc_table, fc_future


if __name__ == "__main__":
    run()