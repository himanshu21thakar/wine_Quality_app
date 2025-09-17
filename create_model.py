import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# --- Download dataset ---
path = kagglehub.dataset_download("yasserh/wine-quality-dataset")
print("Dataset downloaded to:", path)

# Load the CSV file (choose the appropriate one if there are multiple)
df = pd.read_csv(f"{path}/WineQT.csv")

# --- Explore dataset ---
print(df.head())
print(df.info())

# Drop rows with missing values (if any)
df = df.dropna()

# Features and target
X = df.drop(columns=["quality", "Id"])  # 'Id' is not a feature
y = df["quality"]

# Optionally, treat wine quality as classification categories
# Convert quality to binary classification: Good(≥6) vs Not Good(<6)
y = y.apply(lambda x: 1 if x >= 6 else 0)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Standardize Features ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Build SVM Model ---
svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svm_model.fit(X_train, y_train)

# --- Predictions ---
y_pred = svm_model.predict(X_test)

# --- Evaluation ---
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
