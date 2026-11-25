# run_all.py
# -*- coding: utf-8 -*-
import sys, os, time, json, argparse, importlib, traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

STEPS = [
    ("01_lam_sach",   "Bước 1 — Khảo sát & làm sạch dữ liệu (Mục 3.2, 3.3)"),
    ("02_truc_quan",  "Bước 2 — Trực quan hóa (Mục 3.4, Hình 3.1 → 3.7)"),
    ("03_phan_khuc",  "Bước 3 — Phân khúc sản phẩm (Mục 3.5.1)"),
    ("04_du_bao",     "Bước 4 — Dự báo theo tháng (Mục 3.5.2)"),
    ("05_phan_loai",  "Bước 5 — Phân loại nguy cơ tiêu cực (Mục 3.5.3)"),
]


def print_banner():
    print("\n" + "=" * 68)
    print("  PIPELINE PHÂN TÍCH DỮ LIỆU VIECOMREC".center(68))
    print("=" * 68 + "\n")


def print_step_header(idx, total, title):
    print("\n" + "─" * 68)
    print(f"  [{idx}/{total}] {title}")
    print("─" * 68)


def check_prerequisites(step_name):
    if step_name != "01_lam_sach":
        clean_path = os.path.join(BASE_DIR, "data", "clean.csv")
        if not os.path.exists(clean_path):
            raise FileNotFoundError(
                f"Thiếu 'data/clean.csv' — bước '{step_name}' cần dữ liệu này.\n"
                f"→ Chạy 'python 01_lam_sach.py' trước, hoặc bỏ --skip-clean.")


def run_step(idx, total, module_name, title):
    print_step_header(idx, total, title)
    check_prerequisites(module_name)
    t0 = time.time()
    try:
        mod = importlib.import_module(module_name)
        importlib.reload(mod)
        result = mod.run()
        elapsed = time.time() - t0
        print(f"\n✅ Hoàn tất '{module_name}' trong {elapsed:.1f}s")
        return True, elapsed, result
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n❌ LỖI ở '{module_name}' sau {elapsed:.1f}s:")
        print(f"   {type(e).__name__}: {e}")
        traceback.print_exc()
        return False, elapsed, None


def write_summary():
    try:
        from config import SUMMARY, OUT_DIR
        path = os.path.join(OUT_DIR, "tong_hop_so_lieu.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(SUMMARY, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📄 Đã ghi tổng hợp số liệu: {path}")
    except Exception as e:
        print(f"\n⚠️  Không ghi được tổng hợp: {e}")


def main():
    parser = argparse.ArgumentParser(description="Chạy toàn bộ pipeline ViEcomRec")
    parser.add_argument("--skip-clean", action="store_true",
                        help="Bỏ qua bước 01 (dùng data/clean.csv có sẵn)")
    parser.add_argument("--only", nargs="+", type=int, metavar="N",
                        help="Chỉ chạy các bước có số thứ tự cho trước (vd: --only 3 4)")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Tiếp tục chạy các bước sau nếu một bước lỗi")
    args = parser.parse_args()

    print_banner()

    selected = []
    for i, (mod, title) in enumerate(STEPS, start=1):
        if args.only and i not in args.only:
            continue
        if args.skip_clean and mod == "01_lam_sach":
            print(f"⏭  Bỏ qua bước {i} ({mod}) theo --skip-clean")
            continue
        selected.append((i, mod, title))

    if not selected:
        print("Không có bước nào để chạy.")
        return 1

    print(f"Sẽ chạy {len(selected)} bước: "
          f"{', '.join(m for _, m, _ in selected)}\n")

    results = []
    t_start = time.time()
    for idx, mod, title in selected:
        ok, elapsed, _ = run_step(idx, len(STEPS), mod, title)
        results.append((mod, ok, elapsed))
        if not ok and not args.continue_on_error:
            print("\n🛑 Dừng pipeline do có lỗi "
                  "(dùng --continue-on-error để bỏ qua).")
            break

    write_summary()

    total_time = time.time() - t_start
    print("\n" + "=" * 68)
    print("  TỔNG KẾT".center(68))
    print("=" * 68)
    for mod, ok, elapsed in results:
        status = "✅ OK  " if ok else "❌ LỖI "
        print(f"  {status} {mod:<20s}  {elapsed:>6.1f}s")
    print(f"\n  Tổng thời gian: {total_time:.1f}s")
    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"  Kết quả: {n_ok}/{len(results)} bước thành công")
    print("=" * 68)

    if n_ok == len(results):
        print("\n🎉 Tất cả các bước đã chạy xong!")
        print(f"   → Mở 'figures/' để xem biểu đồ")
        print(f"   → Mở 'outputs/' để xem bảng CSV và tong_hop_so_lieu.json")
        return 0
    else:
        print("\n⚠️  Có bước bị lỗi — xem log phía trên để gỡ.")
        return 1


if __name__ == "__main__":
    sys.exit(main())