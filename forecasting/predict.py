import pandas as pd
import numpy as np
from prophet import Prophet
import warnings, json
warnings.filterwarnings("ignore")

# ── Cấu hình lead time theo Shipment Provider ─────────────────────────────
# Dựa trên cột ShipmentProvider trong dataset thực
LEAD_TIME_CONFIG = {
    "UPS":        7,
    "Royal Mail": 5,
    "DHL":        4,
    "FedEx":      3,
    "default":    7
}
SERVICE_LEVEL_Z = 1.65  # 95% service level

def get_lead_time(provider: str) -> int:
    return LEAD_TIME_CONFIG.get(provider, LEAD_TIME_CONFIG["default"])


def forecast_sku(weekly_df: pd.DataFrame, sku_id: str,
                 lead_time: int = 7) -> dict:
    """
    Forecast demand 4 tuần tới (≈30 ngày) cho một SKU.
    Trả về forecast + safety stock + reorder point.
    """
    ts = (weekly_df[weekly_df["StockCode"] == sku_id]
          [["week_start", "net_qty"]]
          .rename(columns={"week_start": "ds", "net_qty": "y"})
          .sort_values("ds"))

    if len(ts) < 8:  # Không đủ data → dùng moving average
        avg_weekly = ts["y"].mean()
        return {
            "sku_id": sku_id,
            "forecast_30d": round(avg_weekly * 4),
            "avg_daily_demand": round(avg_weekly / 7, 2),
            "safety_stock": round(avg_weekly * 0.5),
            "reorder_point": round(avg_weekly * (lead_time / 7) + avg_weekly * 0.5),
            "method": "moving_average",
            "confidence": "low"
        }

    # ── Chạy Prophet ────────────────────────────────────────────────────
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,  # weekly data, không cần
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10
    )
    model.fit(ts)

    future   = model.make_future_dataframe(periods=4, freq="W")
    forecast = model.predict(future)
    future_fc = forecast[forecast["ds"] > ts["ds"].max()].copy()
    future_fc["yhat"] = future_fc["yhat"].clip(lower=0)

    # ── Tính chỉ số tồn kho ─────────────────────────────────────────────
    avg_weekly_demand  = future_fc["yhat"].mean()
    avg_daily_demand   = avg_weekly_demand / 7
    demand_std_weekly  = ts["y"].std()
    lead_time_weeks    = lead_time / 7

    # Safety Stock = Z × σ_weekly × √(lead_time_in_weeks)
    safety_stock  = round(SERVICE_LEVEL_Z * demand_std_weekly * (lead_time_weeks ** 0.5))
    reorder_point = round(avg_daily_demand * lead_time + safety_stock)
    forecast_30d  = round(future_fc["yhat"].sum())

    # Forecast accuracy proxy (MAPE trên 4 tuần cuối historical)
    last_4 = ts.tail(4).copy()
    hist_fc = forecast[forecast["ds"].isin(last_4["ds"])]
    if len(hist_fc) == len(last_4):
        mape = (abs(last_4["y"].values - hist_fc["yhat"].values) /
                (last_4["y"].values + 1e-9)).mean() * 100
    else:
        mape = None

    return {
        "sku_id":           sku_id,
        "forecast_30d":     forecast_30d,
        "avg_daily_demand": round(avg_daily_demand, 2),
        "safety_stock":     safety_stock,
        "reorder_point":    reorder_point,
        "mape":             round(mape, 1) if mape else None,
        "method":           "prophet",
        "confidence":       "high" if (mape or 999) < 20 else "medium"
    }


def run_all_skus(weekly_path: str, raw_path: str) -> pd.DataFrame:
    weekly = pd.read_csv(weekly_path, parse_dates=["week_start"])
    raw    = pd.read_csv(raw_path)

    # Lấy lead time từ ShipmentProvider phổ biến nhất của mỗi SKU
    lead_times = (raw.groupby("StockCode")["ShipmentProvider"]
                     .agg(lambda x: x.mode()[0] if len(x) > 0 else "default"))

    results, skus = [], weekly["StockCode"].unique()
    for i, sku in enumerate(skus):
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(skus)}] forecasting...")
        provider  = lead_times.get(sku, "default")
        lead_time = get_lead_time(provider)
        result    = forecast_sku(weekly, sku, lead_time)
        result["shipment_provider"] = provider
        result["lead_time_days"]    = lead_time
        results.append(result)

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Running forecasts for all 1,000 SKUs...")
    results = run_all_skus("data/weekly_demand.csv",
                           "data/online_retail_dataset.csv")
    results.to_csv("data/forecast_output.csv", index=False)
    print("\nResults summary:")
    print(results[["sku_id","forecast_30d","safety_stock",
                   "reorder_point","mape","confidence"]].head(10))
    print("\nConfidence distribution:")
    print(results["confidence"].value_counts())