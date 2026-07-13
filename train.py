import joblib 
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/iris.csv")

print(df.head())

X = df.drop(columns=["species", "flower_names"])

y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.2f}")

joblib.dump(
    accuracy,
    "models/accuracy.pkl"
)

joblib.dump(
    model, "models/iris_model.pkl"
)

print("Model saved successfully!")

