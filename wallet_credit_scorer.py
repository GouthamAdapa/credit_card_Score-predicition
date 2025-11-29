import json
import pandas as pd
from datetime import datetime
import argparse
from collections import defaultdict

def load_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def preprocess(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['date'] = df['timestamp'].dt.date

    # Extract numeric amount from nested actionData field
    df['amount'] = df['actionData'].apply(lambda x: float(x.get('amount', 0)) if isinstance(x, dict) else 0)

    return df

def feature_engineering(df):
    wallets = defaultdict(lambda: defaultdict(float))
    grouped = df.groupby('userWallet')

    for wallet, group in grouped:
        actions = group['action'].value_counts().to_dict()
        total_actions = len(group)

        deposit_amt = group[group['action'] == 'deposit']['amount'].sum()
        borrow_amt = group[group['action'] == 'borrow']['amount'].sum()
        repay_amt = group[group['action'] == 'repay']['amount'].sum()
        liquidation_calls = actions.get('liquidationcall', 0)

        activity_days = group['date'].nunique()
        repay_ratio = repay_amt / borrow_amt if borrow_amt > 0 else 1
        borrow_to_deposit = borrow_amt / deposit_amt if deposit_amt > 0 else 0
        liquidation_rate = liquidation_calls / total_actions if total_actions > 0 else 0
        activity_score = min(activity_days / 30, 1.0)
        action_score = min(total_actions / 50, 1.0)

        score = (
            0.3 * repay_ratio +
            0.2 * (1 - liquidation_rate) +
            0.2 * (1 - borrow_to_deposit) +
            0.15 * activity_score +
            0.15 * action_score
        ) * 1000
        score = max(0, min(1000, score))  # Clamp between 0 and 1000

        wallets[wallet]['score'] = round(score)
        wallets[wallet]['repay_ratio'] = round(repay_ratio, 2)
        wallets[wallet]['liquidation_rate'] = round(liquidation_rate, 2)
        wallets[wallet]['borrow_to_deposit'] = round(borrow_to_deposit, 2)
        wallets[wallet]['activity_days'] = activity_days
        wallets[wallet]['total_actions'] = total_actions

    return pd.DataFrame.from_dict(wallets, orient='index').reset_index().rename(columns={'index': 'wallet'})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="Path to JSON file")
    parser.add_argument('--output', default="wallet_scores.csv", help="Path to output CSV")
    args = parser.parse_args()

    df = load_data(args.input)
    df = preprocess(df)
    result_df = feature_engineering(df)
    result_df.to_csv(args.output, index=False)
    print(f"✅ Scoring complete. Output saved to {args.output}")

if __name__ == "__main__":
    main()
