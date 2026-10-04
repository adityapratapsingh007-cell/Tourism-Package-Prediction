import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "tourism_project/data/tourism.csv"

if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")

df = pd.read_csv(RAW_PATH)

# Clean inconsistent categorical labels.
df["Gender"] = df["Gender"].replace(
    {"Fe male": "Female", "Fe Male": "Female"}
)
df["MaritalStatus"] = df["MaritalStatus"].replace(
    {"Unmarried": "Single"}
)

# Remove identifier / index columns that are not predictive features.
drop_columns = [
    "CustomerID",
    "UDI",
    "Unnamed: 0",
]
df = df.drop(columns=drop_columns, errors="ignore")

# Separate target from predictors.
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data preparation completed successfully.")
print("Training shape:", Xtrain.shape)
print("Testing shape:", Xtest.shape)
print("Training target distribution:")
print(ytrain.value_counts(normalize=True).round(4))
