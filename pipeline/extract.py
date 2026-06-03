import pandas as pd

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    print(f"Loaded: {df.shape[0]:,} rows | {df['StockCode'].nunique()} SKUs")
    print(f"Date range: {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")
    return df

if __name__ == "__main__":
    df = load_raw("data/online_retail_dataset.csv")
    print(df.head())