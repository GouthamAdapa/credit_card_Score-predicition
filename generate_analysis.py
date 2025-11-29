import pandas as pd

# Load your scored wallet output
df = pd.read_csv("wallet_scores_ml.csv")

# Bin scores into ranges of 100
df['score_band'] = pd.cut(df['score'], bins=[0,100,200,300,400,500,600,700,800,900,1000])

# Count wallets in each score range
band_counts = df['score_band'].value_counts().sort_index()

# Print banded counts
print("\n🎯 Score Distribution:")
for band, count in band_counts.items():
    print(f"{band}: {count} wallets")

# Identify behavior of lowest and highest scorers
low = df[df['score'] <= 300]
high = df[df['score'] >= 700]

print("\n🔻 Low Score Wallets (<300):")
print(low.describe())

print("\n🟢 High Score Wallets (>=700):")
print(high.describe())

# Save score_band summary to file
summary = band_counts.reset_index()
summary.columns = ['score_range', 'wallet_count']
summary.to_csv("score_band_summary.csv", index=False)
print("\n✅ score_band_summary.csv saved.")
