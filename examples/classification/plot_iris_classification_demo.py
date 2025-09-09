# File: examples/classification/iris_classification_demo.py
"""
========================================
Iris Classification with Random Forest
========================================

This example demonstrates how to train a RandomForestClassifier
on the Iris dataset and visualize feature importances.
"""



from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Plot feature importance
plt.bar(iris.feature_names, clf.feature_importances_)
plt.title("Feature Importance in Iris Classification")
plt.show()
