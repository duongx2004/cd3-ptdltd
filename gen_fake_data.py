# gen_fake_data.py
# -*- coding: utf-8 -*-
"""
Sinh dữ liệu giả lập CÓ TÍN HIỆU sản phẩm tốt/kém rõ ràng.
Sản phẩm chia 3 tier chất lượng → mô hình học được sản phẩm nguy cơ.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# 1. BẢNG SẢN PHẨM — CÓ TIER CHẤT LƯỢNG
# ============================================================
N_ITEMS = 300
BRANDS = [f"Brand_{chr(65+i)}" for i in range(15)]
ORIGINS = ["Hàn Quốc", "Việt Nam", "Nhật Bản", "Mỹ", "Pháp"]
SKIN_TYPES = ["Da dầu", "Da khô", "Da hỗn hợp", "Da nhạy cảm", "Mọi loại da"]
VOLUMES = [50, 80, 100, 120, 150, 200]

# ⭐ MỖI SẢN PHẨM CÓ TIER CHẤT LƯỢNG
# tier = "tot"   (60%): P(1-2)=0.05, P(3)=0.10, P(4-5)=0.85
# tier = "tb"    (30%): P(1-2)=0.20, P(3)=0.20, P(4-5)=0.60
# tier = "kem"   (10%): P(1-2)=0.50, P(3)=0.20, P(4-5)=0.30
tiers = np.random.choice(["tot", "tb", "kem"], N_ITEMS, p=[0.60, 0.30, 0.10])
tier_probs = {
    "tot": [0.03, 0.02, 0.10, 0.30, 0.55],
    "tb":  [0.12, 0.08, 0.20, 0.25, 0.35],
    "kem": [0.30, 0.20, 0.20, 0.15, 0.15],
}

items = pd.DataFrame({
    "item_id": [f"SP{i:04d}" for i in range(1, N_ITEMS + 1)],
    "product_name": [f"Sữa rửa mặt {np.random.choice(BRANDS)} số {i}"
                     for i in range(1, N_ITEMS + 1)],
    "brand":  np.random.choice(BRANDS, N_ITEMS),
    "origin": np.random.choice(ORIGINS, N_ITEMS,
                                p=[0.40, 0.25, 0.20, 0.10, 0.05]),
    "skin_type": np.random.choice(SKIN_TYPES, N_ITEMS),
    "volume_ml": np.random.choice(VOLUMES, N_ITEMS),
    "quality_tier": tiers,   # ⭐ CỘT ẨN — chỉ để kiểm tra, KHÔNG đưa vào mô hình
})
items.to_csv(os.path.join(DATA_DIR, "attribute_text_filtering.csv"),
             index=False, encoding="utf-8-sig")

# ============================================================
# 2. POPULARITY PARETO
# ============================================================
popularity = np.random.pareto(1.5, N_ITEMS) + 1.0
popularity = popularity / popularity.sum()

# ============================================================
# 3. BẢNG ĐÁNH GIÁ — ĐIỂM PHỤ THUỘC TIER CỦA SẢN PHẨM
# ============================================================
N_REVIEWS = 60000
N_USERS = 8000
USER_IDS = [f"U{i:06d}" for i in range(1, N_USERS + 1)]

months = pd.period_range("2022-01", "2024-12", freq="M")
growth = np.linspace(0.5, 1.5, len(months))
season = 1 + 0.20 * np.sin(np.arange(len(months)) / 12 * 2 * np.pi)
w = growth * season
w = w / w.sum()

chosen_months = np.random.choice(months, N_REVIEWS, p=w)
day_offsets = [np.random.randint(0, m.days_in_month) for m in chosen_months]
hours = np.random.randint(0, 24, N_REVIEWS)
minutes = np.random.randint(0, 60, N_REVIEWS)

times = [datetime(m.year, m.month, 1) + timedelta(days=int(d))
         + timedelta(hours=int(h), minutes=int(mi))
         for m, d, h, mi in zip(chosen_months, day_offsets, hours, minutes)]

item_ids = np.random.choice(items["item_id"].values, N_REVIEWS, p=popularity)
user_ids = np.random.choice(USER_IDS, N_REVIEWS)

# ⭐ RATING PHỤ THUỘC TIER CỦA SẢN PHẨM — đây là tín hiệu chính
item_tier_map = dict(zip(items["item_id"], items["quality_tier"]))
ratings = np.array([
    np.random.choice([1, 2, 3, 4, 5], p=tier_probs[item_tier_map[iid]])
    for iid in item_ids
])

POS = ["Sản phẩm tốt, da mình hợp lắm",
       "Rửa sạch, không khô, sẽ mua lại",
       "Thơm nhẹ, dùng ổn áp",
       "Giao hàng nhanh, đóng gói cẩn thận",
       "Da mình dầu mà dùng thấy sạch lắm",
       "Giá tốt so với chất lượng",
       "Mua lần 3 rồi, vẫn ưng"]
NEG = ["Da mình bị kích ứng, không hợp",
       "Sản phẩm không giống mô tả",
       "Giao chậm, hộp móp",
       "Mùi quá nồng, khó chịu",
       "Dùng bị khô da",
       "Không sạch sâu như quảng cáo",
       "Mua về bị hết hạn"]
NEU = ["Tạm được",
       "Bình thường, không có gì đặc biệt",
       "Đóng gói ổn",
       "Chưa dùng nên chưa biết"]


def pick_comment(r):
    if r >= 4: return np.random.choice(POS)
    if r <= 2: return np.random.choice(NEG)
    return np.random.choice(NEU)


comments = [pick_comment(r) for r in ratings]

reviews = pd.DataFrame({
    "user_id": user_ids,
    "item_id": item_ids,
    "rating": ratings,
    "time": [t.strftime("%Y-%m-%d %H:%M:%S") for t in times],
    "comment": comments,
})
reviews = reviews.sample(frac=1, random_state=42).reset_index(drop=True)
reviews.to_csv(os.path.join(DATA_DIR, "review.csv"),
               index=False, encoding="utf-8-sig")

# ============================================================
# 4. BÁO CÁO — CHO THẤY TÍN HIỆU ĐÃ ĐỦ MẠNH
# ============================================================
print("=" * 65)
print("ĐÃ SINH DỮ LIỆU GIẢ LẬP (CÓ TIER CHẤT LƯỢNG)")
print("=" * 65)
print(f"  review.csv                        : {len(reviews):>7,} dòng")
print(f"  attribute_text_filtering.csv      : {len(items):>7,} dòng")
print(f"  Tỷ lệ tiêu cực tổng thể           : {(reviews['rating'] <= 2).mean()*100:>6.2f} %")
print()
print("  Phân bố tier chất lượng:")
for t in ["tot", "tb", "kem"]:
    n_items = (items["quality_tier"] == t).sum()
    sub = reviews[reviews["item_id"].isin(
        items[items["quality_tier"] == t]["item_id"])]
    ty_le_neg = (sub["rating"] <= 2).mean() * 100
    print(f"    {t:>4}: {n_items:>3} sản phẩm | "
          f"{len(sub):>6,} đánh giá | "
          f"tỷ lệ tiêu cực: {ty_le_neg:>5.2f}%")
print()
print("  → Tier 'kem' có tỷ lệ tiêu cực cao gấp ~10 lần tier 'tot'")
print("  → Mô hình SẼ học được tín hiệu này từ đặc trưng prior_rate")
print("=" * 65)