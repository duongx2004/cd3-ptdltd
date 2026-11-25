# config.py
# -*- coding: utf-8 -*-
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
FIG_DIR  = os.path.join(BASE_DIR, "figures")

# ========== FILE DỮ LIỆU ==========
INTERACTION_FILES = [
    os.path.join(DATA_DIR, "review.csv"),
]
ITEMS_FILE = os.path.join(DATA_DIR, "attribute_text_filtering.csv")

# ========== TÊN CỘT (None = tự đoán) ==========
COLS = dict(user=None, item=None, rating=None, time=None, text=None)
ITEM_ID_COL = None
ITEM_NAME_COL = None
BRAND_COL = None
ORIGIN_COL = None
ITEM_CAT_COLS = None

# ========== THAM SỐ PHÂN TÍCH ==========
RATING_MIN, RATING_MAX = 1, 5
NEG_THRESHOLD = 2
N_MIN = 30
N_MIN_RISK = 20
TEST_MONTHS = 6
HORIZON = 3
DROP_EDGE_MONTHS = True
TRAIN_FRAC = 0.8
RANDOM_STATE = 42

# ========== TẠO THƯ MỤC ==========
for d in (DATA_DIR, OUT_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

# ========== BIẾN TOÀN CỤC ==========
SUMMARY = {}