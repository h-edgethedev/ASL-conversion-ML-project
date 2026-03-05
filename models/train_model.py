import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = r"models\data\dataset_fixed.csv"
MODEL_PATH = "models/sign_model.joblib"

os.makedirs("models", exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Features = f0..f62
X = df.drop(columns=["label", "person_id"], errors="ignore")
y = df["label"]

# split (keeps hello/yes ratio consistent)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print(f"Accuracy: {acc:.3f}")
print(classification_report(y_test, pred))

joblib.dump(model, MODEL_PATH)
print("Saved model to:", MODEL_PATH)