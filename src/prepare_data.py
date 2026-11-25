"""Stream a RANDOM sample of reczoo/Avazu_x1 straight out of the zip (no full extraction).

Source: https://huggingface.co/datasets/reczoo/Avazu_x1 (originally Kaggle Avazu CTR challenge, 2014).
Raw rows are never committed; run this script to rebuild data/*.parquet.
Files hold 28.3M / 4.0M / 8.1M rows; we keep a uniform random fraction of each (seed 42), stored as int32.
"""
import sys, zipfile, urllib.request, numpy as np, pandas as pd
from pathlib import Path

URL = "https://huggingface.co/datasets/reczoo/Avazu_x1/resolve/main/Avazu_x1.zip"
DATA = Path(__file__).resolve().parent.parent / "data"
ZIP = DATA / "avazu.zip"
FRACTION = {"train": 8_000_000 / 28_300_276, "valid": 1_000_000 / 4_042_897, "test": 2_000_000 / 8_085_794}

def main():
    DATA.mkdir(exist_ok=True)
    if not ZIP.exists():
        print("downloading 625 MB ...")
        urllib.request.urlretrieve(URL, ZIP)
    rng = np.random.default_rng(42)
    z = zipfile.ZipFile(ZIP)
    for split, frac in FRACTION.items():
        name = next(i.filename for i in z.infolist() if i.filename.endswith(f"{split}.csv"))
        parts = []
        with z.open(name) as f:
            for chunk in pd.read_csv(f, chunksize=1_000_000, dtype="int32"):
                parts.append(chunk[rng.random(len(chunk)) < frac])
        df = pd.concat(parts, ignore_index=True)
        df["label"] = df["label"].astype("int8")
        df.to_parquet(DATA / f"{split}.parquet", index=False)
        print(split, df.shape, f"CTR={df.label.mean():.4f}")

if __name__ == "__main__":
    sys.exit(main())
