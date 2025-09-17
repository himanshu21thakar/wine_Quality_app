import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# --- Download dataset ---
path = kagglehub.dataset_download("yasserh/wine-quality-dataset")
df = pd.read_csv(f"{path}/WineQT.csv")

# --- Prepare data ---
df = df.dropna()
X = df.drop(columns=["quality", "Id"])
y = df["quality"].apply(lambda x: 1 if x >= 6 else 0)

# --- Split and scale ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Train SVM model ---
svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svm_model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = svm_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# --- Save the model and scaler ---
joblib.dump(svm_model, "wine_svm_model.pkl")
joblib.dump(scaler, "wine_scaler.pkl")
print("Model and scaler saved as 'wine_svm_model.pkl' and 'wine_scaler.pkl'")
