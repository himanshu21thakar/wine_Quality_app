# 🍷 Wine Quality Classifier  

Predict wine quality using an **SVM (Support Vector Machine)** model trained on the [Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset). This app classifies wine as **Good (≥6)** or **Not Good (<6)** based on its chemical properties.  

---

## 🌐 Live Demo  
🚀 **Try it now:** [Wine Quality App on Streamlit](https://winequalityapp-himanshu.streamlit.app/)  

---

## 🧠 Project Overview  
This project applies machine learning to predict wine quality. The dataset includes 11 physiochemical properties such as acidity, residual sugar, sulphates, and alcohol. The model was trained with **SVM**, scaled with **StandardScaler**, and deployed using **Streamlit** for an interactive experience.  

---

## 📊 Tech Stack  
- **Python 3.10+**  
- **scikit-learn** – Model training and evaluation  
- **joblib** – Model serialization  
- **Streamlit** – Web app interface  
- **NumPy / Pandas** – Data manipulation  

---

## ⚡ Features  
✅ Modern UI with **dynamic CSS**  
✅ Easy-to-use sliders and number inputs  
✅ Real-time predictions of wine quality  
✅ Deployed online via Streamlit  
✅ Simple to retrain or extend with new data  

---

## 🚀 Setup Instructions  

### 1️⃣ Clone the Repository  
git clone https://github.com/your-username/wine-quality-classifier.git
cd wine-quality-classifier

### 2️⃣ Install Requirements
pip install -r requirements.txt

### 3️⃣ Run Locally

Ensure you have the trained model files (wine_svm_model.pkl and wine_scaler.pkl) in the same directory, then:

streamlit run app.py

### 4️⃣ Open in Browser

Go to http://localhost:8501


