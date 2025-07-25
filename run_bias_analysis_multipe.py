import argparse
import pandas as pd
from metrics import bias_risk, eod, paired_ttest

def run_analysis(df, target_attr, subgroup_attrs):
    print(f"\n📌 Analyzing bias for: {target_attr}")
    brisk_max, brisk_mean = bias_risk(df, target_attr, subgroup_attrs)
    eod_max, eod_mean, _ = eod(df, target_attr)
    p_val = paired_ttest(df, target_attr, subgroup_attrs)

    print(f"  brisk*: {brisk_max:.2f}")
    print(f"  brisk : {brisk_mean:.2f}")
    print(f"  EOD max: {eod_max:.2f}")
    print(f"  EOD mean: {eod_mean:.2f}")
    print(f"  Paired t-test p-value: {p_val:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Run bias analysis for one or more attributes.")
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--attributes', type=str, required=True,
                        help='Comma-separated list of target attributes (e.g. isfemale,isold)')
    parser.add_argument('--subgroups', type=str, required=True,
                        help='Comma-separated list of subgroup attributes (e.g. iswhite,isblackhair,...)')

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    target_attributes = [a.strip() for a in args.attributes.split(',')]
    subgroup_attributes = [s.strip() for s in args.subgroups.split(',')]

    for target_attr in target_attributes:
        run_analysis(df, target_attr, subgroup_attributes)

if __name__ == '__main__':
    main()
