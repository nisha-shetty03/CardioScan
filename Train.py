import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---- 1. Load data ----
df = pd.read_csv("heart.csv")

# ---- 2. Encode categorical features, and SAVE the mapping used ----
cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
encoders = {}
for col in cat_cols:
    df[col] = df[col].astype("category")
    encoders[col] = dict(enumerate(df[col].cat.categories))  # {0: 'ASY', 1: 'ATA', ...}
    df[col] = df[col].cat.codes

print("Category encodings used (save these — app.py must match exactly):")
for col, mapping in encoders.items():
    print(f"  {col}: {mapping}")
print()

# ---- 3. Split ----
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- 4. Train (tuned to reduce overfitting vs. the original max_depth=9) ----
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=4,
    min_samples_split=2,
    criterion="entropy",
    random_state=42,
)
model.fit(X_train, y_train)

# ---- 5. Evaluate honestly ----
y_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print(f"Train accuracy: {accuracy_score(y_train, train_pred)*100:.2f}%")
print(f"Test accuracy:  {accuracy_score(y_test, y_pred)*100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

cv_scores = cross_val_score(model, X, y, cv=5)
print(f"5-fold CV accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

# ---- 6. Save model + encoders together ----
joblib.dump(model, "heart_model.pkl")
joblib.dump(encoders, "encoders.pkl")
print("\nSaved heart_model.pkl and encoders.pkl")
