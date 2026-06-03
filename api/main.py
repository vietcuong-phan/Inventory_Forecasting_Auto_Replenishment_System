"""
api/main.py — Inventory Forecast REST API
Chạy: uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# ── Khởi tạo app ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Inventory Forecast API",
    description="Dự báo tồn kho & cảnh báo low-stock",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Đường dẫn file data ───────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY_PATH   = os.path.join(BASE_DIR, "data", "daily_demand.csv")
FORECAST_PATH= os.path.join(BASE_DIR, "data", "forecast_output.csv")
RAW_PATH     = os.path.join(BASE_DIR, "data", "online_retail_dataset.csv")

# ── Load & kiểm tra file ──────────────────────────────────────────────────────
def _check_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File không tồn tại: {path}\n"
            f"Hãy chạy pipeline trước:\n"
            f"  python pipeline/transform.py\n"
            f"  python forecasting/predict.py"
        )

_check_file(DAILY_PATH)
_check_file(FORECAST_PATH)

daily_df    = pd.read_csv(DAILY_PATH,    parse_dates=["date"])
forecast_df = pd.read_csv(FORECAST_PATH)

# ── FIX: forecast_output.csv dùng cột "sku_id", cần rename → "StockCode" ─────
#    predict.py lưu với tên "sku_id", main.py cần đồng nhất về "StockCode"
if "sku_id" in forecast_df.columns and "StockCode" not in forecast_df.columns:
    forecast_df = forecast_df.rename(columns={"sku_id": "StockCode"})

# Kiểm tra cột bắt buộc
required_forecast_cols = {"StockCode", "forecast_30d", "reorder_point", "safety_stock"}
missing = required_forecast_cols - set(forecast_df.columns)
if missing:
    raise ValueError(
        f"forecast_output.csv thiếu các cột: {missing}\n"
        f"Cột hiện có: {list(forecast_df.columns)}"
    )

# ── Tính inventory snapshot từ daily_demand ───────────────────────────────────
#    (Dataset không có bảng tồn kho thực → dùng cumulative net demand làm proxy)
INITIAL_STOCK = 500  # Thay bằng giá trị thực từ ERP nếu có

inventory_df = (
    daily_df
    .groupby(["StockCode", "WarehouseLocation", "Category"])
    .agg(
        total_sold = ("net_qty", "sum"),
        last_30d   = ("net_qty", lambda x: x[
            daily_df.loc[x.index, "date"] >= daily_df["date"].max() - pd.Timedelta(30, "D")
        ].sum())
    )
    .reset_index()
)
inventory_df["quantity_on_hand"] = (
    INITIAL_STOCK - inventory_df["total_sold"]
).clip(lower=0).astype(int)

# ── Merge inventory + forecast ────────────────────────────────────────────────
#    FIX: dùng đúng tên biến forecast_df (không phải forecast)
#    FIX: merge trên "StockCode" (đã rename từ sku_id ở trên)
merged_df = inventory_df.merge(forecast_df, on="StockCode", how="left")

# Phân loại status
def classify_status(row) -> str:
    rp = row.get("reorder_point", 0) or 0
    qty = row["quantity_on_hand"]
    if qty <= rp:
        return "understock"
    elif qty > rp * 3:
        return "overstock"
    return "ok"

merged_df["status"] = merged_df.apply(classify_status, axis=1)

print(f"✅ API loaded: {len(merged_df):,} SKU-Warehouse records")
print(f"   Understock: {(merged_df['status']=='understock').sum()}")
print(f"   OK:         {(merged_df['status']=='ok').sum()}")
print(f"   Overstock:  {(merged_df['status']=='overstock').sum()}")


# ════════════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "message": "Inventory Forecast API is running",
        "docs": "/docs"
    }


@app.get("/forecast/{sku_id}", tags=["Forecast"])
def get_forecast(sku_id: str):
    """Trả về forecast 30 ngày + safety stock + reorder point cho một SKU."""
    row = forecast_df[forecast_df["StockCode"] == sku_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"SKU '{sku_id}' không tìm thấy")
    return row.iloc[0].to_dict()


@app.get("/inventory/status", tags=["Inventory"])
def get_inventory_status(
    category:  str = Query(None, description="Lọc theo category"),
    warehouse: str = Query(None, description="Lọc theo warehouse"),
    status:    str = Query(None, description="understock | ok | overstock"),
    limit:     int = Query(100,  description="Số dòng trả về tối đa"),
):
    """Tổng quan tồn kho toàn bộ SKU, có thể lọc theo category/warehouse/status."""
    df = merged_df.copy()
    if category:  df = df[df["Category"] == category]
    if warehouse: df = df[df["WarehouseLocation"] == warehouse]
    if status:    df = df[df["status"] == status]

    cols = ["StockCode", "Category", "WarehouseLocation",
            "quantity_on_hand", "reorder_point", "forecast_30d", "status"]
    # Chỉ lấy cột tồn tại
    cols = [c for c in cols if c in df.columns]
    return df[cols].head(limit).to_dict("records")


@app.get("/alerts/low-stock", tags=["Alerts"])
def get_low_stock_alerts(
    top_n:     int = Query(20,  description="Số SKU khẩn nhất"),
    warehouse: str = Query(None, description="Lọc theo warehouse"),
    category:  str = Query(None, description="Lọc theo category"),
):
    """Top N SKU cần đặt hàng ngay — sắp xếp theo mức độ thiếu hụt."""
    df = merged_df[merged_df["status"] == "understock"].copy()
    if warehouse: df = df[df["WarehouseLocation"] == warehouse]
    if category:  df = df[df["Category"] == category]

    df["gap"] = (df["reorder_point"] - df["quantity_on_hand"]).clip(lower=0)
    df = df.sort_values("gap", ascending=False).head(top_n)

    cols = ["StockCode", "Category", "WarehouseLocation",
            "quantity_on_hand", "reorder_point", "gap"]
    # Thêm lead_time nếu có
    for opt in ["lead_time_days", "shipment_provider"]:
        if opt in df.columns:
            cols.append(opt)
    cols = [c for c in cols if c in df.columns]
    return df[cols].to_dict("records")


@app.get("/alerts/overstock", tags=["Alerts"])
def get_overstock_alerts(
    top_n:     int = Query(20,  description="Số SKU dư nhiều nhất"),
    warehouse: str = Query(None),
    category:  str = Query(None),
):
    """Top N SKU đang bị tồn kho dư thừa."""
    df = merged_df[merged_df["status"] == "overstock"].copy()
    if warehouse: df = df[df["WarehouseLocation"] == warehouse]
    if category:  df = df[df["Category"] == category]

    df["excess"] = (df["quantity_on_hand"] - df["reorder_point"]).clip(lower=0)
    df = df.sort_values("excess", ascending=False).head(top_n)

    cols = ["StockCode", "Category", "WarehouseLocation",
            "quantity_on_hand", "reorder_point", "excess"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].to_dict("records")


@app.get("/analytics/by-category", tags=["Analytics"])
def get_category_summary():
    """Tổng hợp demand 30 ngày + forecast theo Category."""
    cutoff = daily_df["date"].max() - pd.Timedelta(30, "D")
    demand_30d = (
        daily_df[daily_df["date"] >= cutoff]
        .groupby("Category")["net_qty"]
        .sum()
        .reset_index()
        .rename(columns={"net_qty": "demand_30d"})
    )
    fc_cat = (
        forecast_df.merge(
            daily_df[["StockCode","Category"]].drop_duplicates(),
            on="StockCode", how="left"
        )
        .groupby("Category")["forecast_30d"]
        .sum()
        .reset_index()
    )
    result = demand_30d.merge(fc_cat, on="Category", how="left")
    # Thêm số SKU mỗi category
    sku_count = (
        merged_df.groupby("Category")["StockCode"]
        .nunique()
        .reset_index()
        .rename(columns={"StockCode": "total_skus"})
    )
    result = result.merge(sku_count, on="Category", how="left")
    return result.to_dict("records")


@app.get("/analytics/demand-trend/{sku_id}", tags=["Analytics"])
def get_demand_trend(
    sku_id: str,
    weeks:  int = Query(12, description="Số tuần nhìn lại"),
):
    """Weekly demand trend cho một SKU — dùng để vẽ line chart."""
    df = daily_df[daily_df["StockCode"] == sku_id].copy()
    if df.empty:
        raise HTTPException(status_code=404, detail=f"SKU '{sku_id}' không tìm thấy")

    trend = (
        df.set_index("date")["net_qty"]
          .resample("W").sum()
          .tail(weeks)
          .reset_index()
          .rename(columns={"date": "week", "net_qty": "demand"})
    )
    trend["week"] = trend["week"].dt.strftime("%Y-%m-%d")
    return trend.to_dict("records")


@app.get("/analytics/warehouse-summary", tags=["Analytics"])
def get_warehouse_summary():
    """Tổng hợp tình trạng tồn kho theo Warehouse."""
    summary = (
        merged_df
        .groupby("WarehouseLocation")
        .agg(
            total_skus  = ("StockCode", "nunique"),
            understock  = ("status", lambda x: (x == "understock").sum()),
            ok          = ("status", lambda x: (x == "ok").sum()),
            overstock   = ("status", lambda x: (x == "overstock").sum()),
        )
        .reset_index()
    )
    return summary.to_dict("records")


@app.get("/meta/filters", tags=["Meta"])
def get_filter_options():
    """Trả về danh sách category và warehouse để dùng cho dropdown UI."""
    return {
        "categories": sorted(merged_df["Category"].dropna().unique().tolist()),
        "warehouses": sorted(merged_df["WarehouseLocation"].dropna().unique().tolist()),
        "statuses":   ["understock", "ok", "overstock"],
    }