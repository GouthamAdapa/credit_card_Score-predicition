import json

with open("user-transactions.json", "r") as f:
    data = json.load(f)

# Show first record
print(json.dumps(data[0], indent=2))

# Print all top-level keys
print("Available keys in each record:", list(data[0].keys()))