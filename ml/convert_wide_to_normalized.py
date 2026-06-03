"""Convert wide feature CSV to a normalized parquet expected by training pipeline.

This script will:
- read `DataSet.csv`
- use the first column as `account_id` if no explicit `account_id` column
- keep columns starting with 'F' plus `account_id` and `F3924` label if present
- write `data/processed/normalized.parquet`

Usage:
  python -m ml.convert_wide_to_normalized --in DataSet.csv --out data/processed/normalized.parquet
"""
import argparse
import os
import pandas as pd


def convert(in_csv: str, out_parquet: str):
    df = pd.read_csv(in_csv, dtype=str)
    cols = list(df.columns)
    # if first column header is empty or not 'account_id', treat it as account id
    first = cols[0]
    if first != 'account_id':
        df = df.rename(columns={first: 'account_id'})
    # coerce account_id to str
    df['account_id'] = df['account_id'].astype(str)

    # keep F* columns and account_id
    fcols = [c for c in df.columns if str(c).startswith('F')]
    keep = ['account_id'] + fcols
    df_sel = df.loc[:, [c for c in keep if c in df.columns]]

    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    # attempt to convert numeric-looking columns
    for c in df_sel.columns:
        if c != 'account_id':
            df_sel[c] = pd.to_numeric(df_sel[c], errors='coerce')

    df_sel.to_parquet(out_parquet, index=False)
    return out_parquet


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='in_csv', required=True)
    p.add_argument('--out', dest='out_parquet', required=True)
    args = p.parse_args()
    print(convert(args.in_csv, args.out_parquet))


if __name__ == '__main__':
    _cli()
