# Customer-churn-prediction
A Machine Learning web app built with **Streamlit** that predicts whether a customer will churn or stay.   The model is trained using a cleaned telecom-style dataset and saved via **Pickle**.

---


## 🚀 Features
- Predicts customer churn in real time  
- Data cleaned & preprocessed using Pandas  
- One-hot encoding via `get_dummies()`  
- Model: Random Forest Classifier  
- Interactive Streamlit interface

---

## 🧠 Tech Stack
**Python**, **Streamlit**, **scikit-learn**, **pandas**, **numpy**, **pickle**

---


## ⚙️ Setup
```bash
pip install -r requirements.txt
python model_train.py
streamlit run app.py

💡 Usage

Enter customer details in the web app

Click Predict Churn

View result:

⚠️ Likely to Churn

✅ Likely to Stay

📊 Model Info

Algorithm: Random Forest

Accuracy: ~85–90%

Evaluation: Accuracy, Precision, Recall, F1-score

👨‍💻 Author

Sparsh Kumar
