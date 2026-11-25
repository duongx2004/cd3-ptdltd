# 01_lam_sach.py
# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import numpy as np
from config import (INTERACTION_FILES, ITEMS_FILE, COLS, ITEM_ID_COL, ITEM_NAME_COL,
                    BRAND_COL, ORIGIN_COL, ITEM_CAT_COLS, RATING_MIN, RATING_MAX,
                    NEG_THRESHOLD, DATA_DIR, OUT_DIR, SUMMARY)
from utils import read_any, parse_time, clean_text, step


def run():
    # ---------- 1. Nạp dữ liệu ----------
    raw = pd.concat([read_any(p) for p in INTERACTION_FILES], ignore_index=True)
    items_raw = read_any(ITEMS_FILE) if ITEMS_FILE else None
    print("Bảng đánh giá:", raw.shape)

    # ---------- 2. Đoán tên cột ----------
    KEYS = {
        "rating": ["rating", "star", "score", "diem"],
        "time":   ["time", "date", "created", "ngay"],
        "user":   ["user", "customer", "reviewer", "author", "nguoi"],
        "item":   ["item_id", "itemid", "product_id", "productid", "item", "product", "sp"],
        "text":   ["comment", "review_text", "content", "text", "review"],
    }
    used = {v for v in COLS.values() if v}
    for field in ["rating", "time", "user", "item", "text"]:
        if COLS[field]:
            continue
        for c in raw.columns:
            lc = str(c).lower()
            if c in used or (any(x in lc for x in ("name", "title")) and field == "item"):
                continue
            if any(k in lc for k in KEYS[field]):
                COLS[field] = c
                used.add(c)
                break

    miss = [k for k in ("user", "item", "rating", "time") if not COLS[k]]
    if miss:
        raise RuntimeError(f"Không đoán được cột: {miss}. Điền COLS trong config.py.")

    df = raw.rename(columns={v: k for k, v in COLS.items() if v})
    if "text" not in df.columns:
        df["text"] = np.nan

    # ---------- 3. Bảng thuộc tính sản phẩm ----------
    item_id_col = ITEM_ID_COL   # biến cục bộ, tránh UnboundLocalError
    if items_raw is not None:
        if item_id_col is None:
            item_id_col = (COLS["item"] if COLS["item"] in items_raw.columns
                           else next((c for c in items_raw.columns
                                      if "id" in str(c).lower()), None))
        if item_id_col is None:
            raise RuntimeError("Không tìm được cột mã sản phẩm trong ITEMS_FILE.")
        item_tbl = (items_raw.rename(columns={item_id_col: "item"})
                    .drop_duplicates("item").copy())
    else:
        item_tbl = df.drop_duplicates("item").copy()

    name_col = (ITEM_NAME_COL if ITEM_NAME_COL in item_tbl.columns else
                next((c for c in item_tbl.columns
                      if any(k in str(c).lower() for k in ("name", "title", "ten_"))
                      and "brand" not in str(c).lower()), None))

    def find_col(tbl, preset, keys):
        if preset and preset in tbl.columns:
            return preset
        return next((c for c in tbl.columns
                     if any(k in str(c).lower() for k in keys)), None)

    brand_col  = find_col(item_tbl, BRAND_COL,  ["brand", "thuong_hieu", "thuonghieu", "hang"])
    origin_col = find_col(item_tbl, ORIGIN_COL, ["origin", "xuat_xu", "xuatxu", "country", "nuoc"])

    if ITEM_CAT_COLS is not None:
        item_cat_cols = [c for c in ITEM_CAT_COLS if c in item_tbl.columns]
    else:
        skip = {"item", "user", "rating", "time", "text", name_col}
        item_cat_cols = []
        for c in item_tbl.columns:
            if c in skip:
                continue
            if (pd.api.types.is_numeric_dtype(item_tbl[c])
                    and item_tbl[c].nunique() > 50):
                continue
            s = item_tbl[c].dropna().astype(str)
            if s.empty:
                continue
            if 2 <= s.nunique() <= 300 and s.str.len().mean() <= 60:
                item_cat_cols.append(c)

    print("Tên sản phẩm:", name_col, "| Thương hiệu:", brand_col,
          "| Xuất xứ:", origin_col)
    print("Cột thuộc tính dùng cho mô hình:", item_cat_cols)

    # ---------- 4. Làm sạch ----------
    log = []
    d = df.copy()
    d["rating"] = pd.to_numeric(d["rating"], errors="coerce")
    d["time"] = parse_time(d["time"])
    d = step("0. Dữ liệu gốc", d, log)
    d = step("1. Loại thiếu user/item/điểm/thời gian",
             d.dropna(subset=["user", "item", "rating", "time"]), log)
    d = step("2. Loại trùng lặp",
             d.drop_duplicates(subset=["user", "item", "rating", "time", "text"]), log)
    d = step(f"3. Giữ điểm {RATING_MIN}-{RATING_MAX}",
             d[d["rating"].between(RATING_MIN, RATING_MAX)], log)
    d = step("4. Giữ thời gian hợp lệ (2010 → nay)",
             d[(d["time"] >= "2010-01-01") & (d["time"] <= pd.Timestamp.now())], log)

    # ---------- 5. Nối thuộc tính ----------
    attr_cols = [c for c in item_cat_cols if c in item_tbl.columns]
    for c in {brand_col, origin_col} - {None}:
        if c not in attr_cols:
            attr_cols.append(c)
    attr = item_tbl[["item"] + attr_cols].copy()
    for c in attr_cols:
        if not pd.api.types.is_numeric_dtype(attr[c]) or attr[c].nunique() <= 300:
            attr[c] = clean_text(attr[c])
    if name_col:
        attr[name_col] = item_tbl[name_col].values
    if items_raw is not None:
        keep = [c for c in attr.columns if c == "item" or c not in d.columns]
        d = d.merge(attr[keep], on="item", how="left")
        for c in keep:
            if c != "item" and not pd.api.types.is_numeric_dtype(d[c]):
                d[c] = d[c].fillna("Không rõ")
        step("5. Nối thuộc tính sản phẩm", d, log)
    else:
        for c in attr_cols:
            d[c] = clean_text(d[c])
        step("5. Chuẩn hóa thuộc tính (có sẵn)", d, log)

    # ---------- 6. Tạo biến ----------
    d["month"] = d["time"].dt.to_period("M").dt.to_timestamp()
    d["month_num"] = d["time"].dt.month
    d["weekday"] = d["time"].dt.dayofweek
    d["hour"] = d["time"].dt.hour
    d["text_len"] = d["text"].astype("string").str.len()
    d["neg"] = (d["rating"] <= NEG_THRESHOLD).astype(int)
    step("6. Tạo biến tháng/thứ/độ dài/nhãn tiêu cực", d, log)

    df_clean = d.reset_index(drop=True)

    # ---------- 7. Lưu ----------
    pd.DataFrame(log).to_csv(
        os.path.join(OUT_DIR, "bang_3_3_nhat_ky_lam_sach.csv"),
        index=False, encoding="utf-8-sig")
    df_clean.to_csv(os.path.join(DATA_DIR, "clean.csv"),
                    index=False, encoding="utf-8-sig")

    meta = {
        "COLS": COLS,
        "ITEM_ID_COL": item_id_col,
        "ITEM_CAT_COLS": item_cat_cols,
        "BRAND_COL": brand_col,
        "ORIGIN_COL": origin_col,
        "NAME_COL": name_col,
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    SUMMARY["lam_sach"] = dict(
        so_dong_sau=int(len(df_clean)),
        giu_lai_pct=round(len(df_clean) / len(raw) * 100, 2),
        so_nguoi_dung=int(df_clean["user"].nunique()),
        so_san_pham=int(df_clean["item"].nunique()),
        ty_le_tieu_cuc_pct=round(float(df_clean["neg"].mean() * 100), 2),
        diem_tb=round(float(df_clean["rating"].mean()), 3),
        diem_trung_vi=float(df_clean["rating"].median()),
        tu_thang=str(df_clean["month"].min().date()),
        den_thang=str(df_clean["month"].max().date()),
    )
    print(f"\n✅ Đã lưu data/clean.csv ({len(df_clean):,} dòng)")
    print("✅ Đã lưu outputs/metadata.json")
    return df_clean


if __name__ == "__main__":
    run()