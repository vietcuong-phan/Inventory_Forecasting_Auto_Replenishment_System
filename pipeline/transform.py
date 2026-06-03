import pandas as pd
import numpy as np


def clean_and_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input : raw transactions (49,782 rows)
    Output: daily net demand per SKU × Warehouse (aggregated, all dates filled)
    """

    # ── 1. Loại bỏ UnitPrice âm (anomaly, không phải return) ─────────────
    before = len(df)
    df = df[df["UnitPrice"] > 0].copy()
    print(f"Removed {before - len(df)} rows with negative UnitPrice")

    # ── 2. Fill null WarehouseLocation bằng mode theo Category ───────────
    mode_by_cat = (
        df.dropna(subset=["WarehouseLocation"])
          .groupby("Category")["WarehouseLocation"]
          .agg(lambda x: x.mode()[0])
    )
    df["WarehouseLocation"] = df.apply(
        lambda r: mode_by_cat.get(r["Category"], "Unknown")
                  if pd.isna(r["WarehouseLocation"]) else r["WarehouseLocation"],
        axis=1
    )

    # ── 3. Tính NET demand: Quantity < 0 là hàng trả về ──────────────────
    df["date"] = df["InvoiceDate"].dt.normalize()  # bỏ giờ, chỉ lấy ngày

    # ── 4. Aggregate: tổng net demand theo ngày × SKU × Warehouse ────────
    daily = (
        df.groupby(["date", "StockCode", "WarehouseLocation", "Category"])
          .agg(
              net_qty         = ("Quantity", "sum"),
              gross_qty       = ("Quantity", lambda x: x[x > 0].sum()),
              return_qty      = ("Quantity", lambda x: abs(x[x < 0].sum())),
              revenue         = ("UnitPrice", lambda x: (x * df.loc[x.index, "Quantity"]).sum()),
              num_transactions= ("InvoiceNo", "nunique")
          )
          .reset_index()
    )

    # ── 5. Fill ngày không có giao dịch = 0 ──────────────────────────────
    #
    # FIX: Lỗi cũ dùng vòng lặp theo SKU đơn lẻ → một SKU có thể ở nhiều
    # Warehouse → set_index("date") bị duplicate index → reindex lỗi.
    #
    # Giải pháp: duyệt theo cặp (StockCode, WarehouseLocation) để đảm bảo
    # index "date" là duy nhất trong mỗi nhóm.
    # ─────────────────────────────────────────────────────────────────────

    date_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")

    NUM_COLS = ["net_qty", "gross_qty", "return_qty", "revenue", "num_transactions"]
    STR_COLS = ["StockCode", "WarehouseLocation", "Category"]

    # Lấy danh sách các cặp (SKU, Warehouse, Category) duy nhất
    groups = (
        daily[["StockCode", "WarehouseLocation", "Category"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    filled_rows = []
    total = len(groups)

    for i, row in groups.iterrows():
        sku = row["StockCode"]
        wh  = row["WarehouseLocation"]
        cat = row["Category"]

        if (i + 1) % 500 == 0:
            print(f"  Filling dates: {i + 1}/{total} groups...")

        # Lọc đúng cặp (SKU, Warehouse) → index "date" luôn unique
        mask = (
            (daily["StockCode"] == sku) &
            (daily["WarehouseLocation"] == wh)
        )
        group_df = daily[mask].set_index("date")

        # Kiểm tra phòng ngừa: nếu vẫn còn duplicate thì aggregate trước
        if group_df.index.duplicated().any():
            group_df = group_df.groupby(level="date")[NUM_COLS].sum()
            group_df["StockCode"]         = sku
            group_df["WarehouseLocation"] = wh
            group_df["Category"]          = cat

        # Reindex toàn bộ ngày trong khoảng
        full = group_df.reindex(date_range)

        # Fill cột số = 0
        full[NUM_COLS] = full[NUM_COLS].fillna(0)

        # Fill cột string = giá trị cố định của nhóm này
        full["StockCode"]         = sku
        full["WarehouseLocation"] = wh
        full["Category"]          = cat

        full.index.name = "date"
        filled_rows.append(full.reset_index())

    daily_full = pd.concat(filled_rows, ignore_index=True)

    # ── 6. Clip net_qty âm còn lại về 0 (ngày return > sold) ─────────────
    daily_full["net_qty"] = daily_full["net_qty"].clip(lower=0)

    # ── 7. Đảm bảo kiểu dữ liệu đúng ────────────────────────────────────
    for col in NUM_COLS:
        daily_full[col] = pd.to_numeric(daily_full[col], errors="coerce").fillna(0)
    daily_full["num_transactions"] = daily_full["num_transactions"].astype(int)

    print(f"\nClean data: {len(daily_full):,} rows")
    print(f"SKUs: {daily_full['StockCode'].nunique()} | "
          f"Warehouses: {daily_full['WarehouseLocation'].nunique()} | "
          f"Date range: {daily_full['date'].min().date()} → {daily_full['date'].max().date()}")
    return daily_full


def aggregate_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo bảng weekly demand — dùng cho SKU thưa giao dịch.
    Prophet hoạt động tốt hơn với weekly khi daily quá sparse.
    """
    weekly = (
        daily_df
        .groupby([
            pd.Grouper(key="date", freq="W-MON"),
            "StockCode", "WarehouseLocation", "Category"
        ])
        .agg(
            net_qty = ("net_qty", "sum"),
            revenue = ("revenue", "sum")
        )
        .reset_index()
        .rename(columns={"date": "week_start"})
    )
    return weekly


if __name__ == "__main__":
    from extract import load_raw

    df_raw = load_raw("data/online_retail_dataset.csv")

    print("\n── Step 1: clean & aggregate ──")
    daily = clean_and_aggregate(df_raw)

    print("\n── Step 2: aggregate weekly ──")
    weekly = aggregate_weekly(daily)
    print(f"Weekly data: {len(weekly):,} rows")

    print("\n── Step 3: saving... ──")
    daily.to_csv("data/daily_demand.csv", index=False)
    weekly.to_csv("data/weekly_demand.csv", index=False)
    print("Saved: data/daily_demand.csv")
    print("Saved: data/weekly_demand.csv")

    print("\nWeekly sample (SKU_1044):")
    print(weekly[weekly["StockCode"] == "SKU_1044"].head(8))