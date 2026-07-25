import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Load dataset

df = pd.read_csv(
    "data/creditcard.csv"
)


print("Original shape:")
print(df.shape)


# Separate features and labels

X = df.drop(
    "Class",
    axis=1
)

y = df["Class"]


# Remove Time
# It is a timestamp, not a useful transaction feature

X = X.drop(
    "Time",
    axis=1
)


# Normalize features

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# Train/test split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training data:")
print(X_train.shape)

print("Test data:")
print(X_test.shape)


print("Fraud examples:")
print(y_train.sum())