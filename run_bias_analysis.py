import argparse
import pandas as pd
from metrics import bias_risk, eod, paired_ttest

def main():
    parser = argparse.ArgumentParser(description="Run bias analysis metrics.")
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV with scores and attributes')
    parser.add_argument('--attribute', type=str, required=True, help='Target attribute to evaluate (e.g. isfemale)')
    parser.add_argument('--subgroups', type=str, required=True, help='Comma-separated list of other binary attributes')

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    gatt = args.attribute
    sgatt = args.subgroups.split(',')

    print(f"Running bias analysis for attribute: {gatt}\n")

    brisk_max, brisk_mean = bias_risk(df, gatt, sgatt)
    print(f"brisk*: {brisk_max}")
    print(f"brisk: {brisk_mean}")

    eod_max, eod_mean, eod_std = eod(df, gatt)
    print(f"EOD max: {eod_max}")
    print(f"EOD mean: {eod_mean}")

    p_value = paired_ttest(df, gatt, sgatt)
    print(f"Paired t-test p-value: {p_value}")

if __name__ == '__main__':
    main()
