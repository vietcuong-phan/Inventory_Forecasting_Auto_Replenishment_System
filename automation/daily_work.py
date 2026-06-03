"""
automation/daily_job.py — Chạy pipeline tự động hàng ngày
Lệnh chạy thủ công : python automation/daily_job.py
Lệnh chạy scheduler: python automation/daily_job.py --schedule
"""

import sys
import os
import argparse

# ── FIX: Thêm project root vào sys.path ──────────────────────────────────────
# __file__ = .../inventory_forecast/automation/daily_job.py
# ROOT_DIR = .../inventory_forecast/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Bây giờ Python tìm được pipeline, forecasting, api
from pipeline.extract import load_raw
from pipeline.transform import clean_and_aggregate, aggregate_weekly
from forecasting.predict import run_all_skus

import smtplib
import requests
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT_DIR, ".env"))

# ── Đường dẫn file data ───────────────────────────────────────────────────────
DATA_DIR      = os.path.join(ROOT_DIR, "data")
RAW_PATH      = os.path.join(DATA_DIR, "online_retail_dataset.csv")
DAILY_PATH    = os.path.join(DATA_DIR, "daily_demand.csv")
WEEKLY_PATH   = os.path.join(DATA_DIR, "weekly_demand.csv")
FORECAST_PATH = os.path.join(DATA_DIR, "forecast_output.csv")

API_BASE = os.getenv("API_BASE", "http://localhost:8000")


# ── Các bước pipeline ─────────────────────────────────────────────────────────

def step_etl() -> bool:
    print("\n▶ [1/3] Running ETL pipeline...")
    try:
        df_raw = load_raw(RAW_PATH)
        daily  = clean_and_aggregate(df_raw)
        weekly = aggregate_weekly(daily)
        daily.to_csv(DAILY_PATH,   index=False)
        weekly.to_csv(WEEKLY_PATH, index=False)
        print(f"   ✅ ETL done: {len(daily):,} daily rows | {len(weekly):,} weekly rows")
        return True
    except Exception as e:
        print(f"   ❌ ETL failed: {e}")
        return False


def step_forecast() -> bool:
    print("\n▶ [2/3] Running forecasts...")
    try:
        results = run_all_skus(WEEKLY_PATH, RAW_PATH)
        results.to_csv(FORECAST_PATH, index=False)
        print(f"   ✅ Forecast done: {len(results)} SKUs")
        return True
    except Exception as e:
        print(f"   ❌ Forecast failed: {e}")
        return False


def step_alert() -> bool:
    print("\n▶ [3/3] Sending alerts...")
    try:
        resp = requests.get(
            f"{API_BASE}/alerts/low-stock",
            params={"top_n": 15},
            timeout=10
        )
        resp.raise_for_status()
        alerts = resp.json()
        print(f"   Found {len(alerts)} low-stock alerts")
        if alerts:
            send_alert_email(alerts)
        else:
            print("   ℹ️  No alerts to send")
        return True
    except requests.ConnectionError:
        print(f"   ⚠️  API chưa chạy tại {API_BASE} — bỏ qua gửi email")
        print(f"      Hãy chạy: uvicorn api.main:app --port 8000")
        return False
    except Exception as e:
        print(f"   ❌ Alert step failed: {e}")
        return False


def send_alert_email(alerts: list):
    email_from = os.getenv("EMAIL_FROM")
    email_to   = os.getenv("EMAIL_TO")
    email_pass = os.getenv("EMAIL_APP_PASS")

    if not all([email_from, email_to, email_pass]):
        print("   ⚠️  Thiếu EMAIL_FROM / EMAIL_TO / EMAIL_APP_PASS trong .venv")
        print("      Bỏ qua gửi email — top 5 alerts:")
        for a in alerts[:5]:
            print(f"      • {a.get('StockCode','?')} | tồn: {a.get('quantity_on_hand','?')} | thiếu: {a.get('gap','?')}")
        return

    rows_html = "".join([f"""
      <tr>
        <td>{a.get('StockCode','')}</td>
        <td>{a.get('Category','')}</td>
        <td>{a.get('WarehouseLocation','')}</td>
        <td style='color:red;font-weight:bold'>{a.get('quantity_on_hand','')}</td>
        <td>{a.get('reorder_point','')}</td>
        <td style='color:orange;font-weight:bold'>{a.get('gap','')}</td>
      </tr>""" for a in alerts])

    html = f"""
    <h2>⚠️ Cảnh báo tồn kho thấp — {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}</h2>
    <p><b>{len(alerts)} SKU</b> đang dưới reorder point và cần đặt hàng ngay:</p>
    <table border='1' cellpadding='6' cellspacing='0'
           style='border-collapse:collapse;font-family:sans-serif;font-size:13px'>
      <tr style='background:#f0f0f0'>
        <th>SKU</th><th>Category</th><th>Warehouse</th>
        <th>Tồn kho</th><th>Reorder Point</th><th>Còn thiếu</th>
      </tr>
      {rows_html}
    </table>
    <p style='color:#666;font-size:12px'>Email tự động từ Inventory Forecast System</p>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[INVENTORY ALERT] {len(alerts)} SKU cần đặt hàng — {pd.Timestamp.now().date()}"
    msg["From"]    = email_from
    msg["To"]      = email_to
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(email_from, email_pass)
        s.sendmail(email_from, email_to, msg.as_string())
    print(f"   ✅ Email sent to {email_to}")


# ── Hàm chạy toàn bộ pipeline ─────────────────────────────────────────────────

def run_full_pipeline():
    print(f"\n{'='*50}")
    print(f"  Daily pipeline — {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}")
    print(f"{'='*50}")
    ok_etl      = step_etl()
    ok_forecast = step_forecast() if ok_etl      else False
    ok_alert    = step_alert()    if ok_forecast  else False
    print(f"\n{'='*50}")
    status = "✅ ALL DONE" if all([ok_etl, ok_forecast, ok_alert]) else "⚠️  DONE WITH WARNINGS"
    print(f"  {status}")
    print(f"  ETL: {'✅' if ok_etl else '❌'}  |  Forecast: {'✅' if ok_forecast else '❌'}  |  Alert: {'✅' if ok_alert else '⚠️'}")
    print(f"{'='*50}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inventory Forecast Daily Job")
    parser.add_argument(
        "--schedule", action="store_true",
        help="Chạy theo lịch 6:30 AM mỗi ngày. Không có flag thì chạy 1 lần ngay."
    )
    args = parser.parse_args()

    if args.schedule:
        try:
            from apscheduler.schedulers.blocking import BlockingScheduler
        except ImportError:
            print("❌ Thiếu apscheduler. Cài: pip install apscheduler")
            sys.exit(1)

        scheduler = BlockingScheduler(timezone="Asia/Ho_Chi_Minh")
        scheduler.add_job(run_full_pipeline, "cron", hour=6, minute=30)
        print("🕐 Scheduler started — pipeline chạy lúc 06:30 mỗi ngày")
        print("   Nhấn Ctrl+C để dừng\n")
        try:
            scheduler.start()
        except KeyboardInterrupt:
            print("\nScheduler stopped.")
    else:
        run_full_pipeline()
