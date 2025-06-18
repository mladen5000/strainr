import pandas as pd
import sys

def main(parquet_path):
    try:
        df = pd.read_parquet(parquet_path)
        print(f"DataFrame shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        print("Index (K-mers):", df.index.name, df.index.tolist())
        print("Head:")
        print(df.head())
    except Exception as e:
        print(f"Error reading or processing Parquet file {parquet_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python inspect_parquet.py <path_to_parquet_file>")
