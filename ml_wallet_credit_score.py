import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import argparse
from collections import defaultdict

def load_and_preprocess(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['date'] = df['timestamp'].dt.date
    df['amount'] = df['actionData'].apply(lambda x: float(x.get('amount', 0)) if isinstance(x, dict) else 0)
    return df

def feature_engineering(df):
    wallets = defaultdict(dict)
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

        wallets[wallet] = {
            'repay_ratio': repay_ratio,
            'liquidation_rate': liquidation_rate,
            'borrow_to_deposit': borrow_to_deposit,
            'activity_days': activity_days,
            'total_actions': total_actions,
            'activity_score': activity_score,
            'action_score': action_score
        }

    return pd.DataFrame.from_dict(wallets, orient='index').reset_index().rename(columns={'index': 'wallet'})

def create_labels(df):
    # Define "good" = repay_ratio > 0.9, low liquidation, moderate borrowing
    df['label'] = (
        (df['repay_ratio'] > 0.9) &
        (df['liquidation_rate'] < 0.05) &
        (df['borrow_to_deposit'] < 1.0)
    ).astype(int)
    return df

def train_model(df):
    features = ['repay_ratio', 'liquidation_rate', 'borrow_to_deposit', 'activity_days', 'total_actions']
    X = df[features]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

    # Predict probabilities for scoring
    df['score'] = (model.predict_proba(X)[:, 1] * 1000).astype(int)
    return model, df

def plot_distributions(df):
    plt.figure(figsize=(10, 5))
    sns.histplot(df['score'], bins=30, kde=True)
    plt.title('Wallet Credit Score Distribution')
    plt.xlabel('Score (0-1000)')
    plt.ylabel('Wallet Count')
    plt.grid()
    plt.tight_layout()
    plt.savefig("score_distribution.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df[['repay_ratio', 'borrow_to_deposit', 'liquidation_rate']])
    plt.title('Behavior Feature Distributions')
    plt.tight_layout()
    plt.savefig("feature_boxplot.png")
    plt.show()

def save_output(df, output_csv):
    df[['wallet', 'score', 'label']].to_csv(output_csv, index=False)
    print(f"\n✅ Wallet scores saved to: {output_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="Path to JSON file")
    parser.add_argument('--output', default="wallet_scores_ml.csv", help="Path to output CSV")
    args = parser.parse_args()

    print("🔄 Loading & preprocessing...")
    raw_df = load_and_preprocess(args.input)
    print("🔬 Feature engineering...")
    feature_df = feature_engineering(raw_df)
    print("🏷️ Creating heuristic labels...")
    labeled_df = create_labels(feature_df)
    print("🤖 Training ML model...")
    model, scored_df = train_model(labeled_df)
    print("📊 Plotting score distribution...")
    plot_distributions(scored_df)
    save_output(scored_df, args.output)

if __name__ == "__main__":
    main()
