import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb

# Load dataset
df = pd.read_csv("TRAIN.csv").dropna()
X = df.drop("FAKE", axis=1)
y = df["FAKE"]

if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVC": SVC(),
    "AdaBoost": AdaBoostClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss')

}

train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
results = {name: [] for name in classifiers}

for size in train_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, random_state=42)
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name].append(acc)

# Plot learning curves
plt.figure(figsize=(12, 8))
for name, scores in results.items():
    plt.plot([int(s*100) for s in train_sizes], scores, marker='o', label=name)
plt.xlabel("Training Size (%)")
plt.ylabel("Accuracy")
plt.title("Learning Curves for Different Classifiers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
