# 📧 Spam Email Detection using Deep Learning (LSTM)

This project is an **end-to-end Spam Email Detection system** built using **Natural Language Processing (NLP)** and a **Deep Learning LSTM model**.
It also includes a **Streamlit web application** for real-time spam classification.

---

## 🚀 Features

* Spam vs Not Spam email classification
* Text preprocessing (cleaning, stopword removal, tokenization)
* Bidirectional LSTM deep learning model
* EarlyStopping & Learning Rate Scheduling
* Streamlit-based interactive web app
* Clean, professional project structure

---

## 🧠 Tech Stack

* Python
* TensorFlow / Keras
* Scikit-learn
* NLP
* Streamlit

---

## 📂 Project Structure

```
Spam-Email-Detection/
│
├── app.py                     # Streamlit application
├── notebooks/
│   └── training.ipynb         # Model training notebook
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
├── .gitignore                 # Ignore generated model files
└── model/                     # Generated locally (not pushed)
    ├── spam_model.keras
    └── tokenizer.pkl
```

---

## 🔁 Workflow

```
Email Text
   ↓
Text Cleaning
   ↓
Tokenization
   ↓
Padding
   ↓
BiLSTM Model
   ↓
Spam / Not Spam
```

---

## 🏗️ Model Architecture

* Embedding Layer
* Bidirectional LSTM Layer
* Dense Layer (ReLU)
* Dropout
* Output Layer (Sigmoid)

**Loss Function:** Binary Crossentropy
**Optimizer:** Adam

---

## ⚠️ Model Files (Important)

Trained model and tokenizer files are **not included in this repository** due to GitHub file size limits.

They are generated **locally** after training.

### To generate model files:

1. Open `notebooks/training.ipynb`
2. Run all cells from top to bottom
3. This will create:

   ```
   model/
   ├── spam_model.keras
   └── tokenizer.pkl
   ```

---

## 🌐 Streamlit Web App

The Streamlit app allows users to:

* Enter email text
* Get real-time spam prediction
* View prediction confidence

### ▶️ Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

⚠️ Make sure you **train the model first**, otherwise the app will prompt you to do so.

---

## 📈 Applications

* Email spam filtering
* SMS spam detection
* Phishing detection
* Customer support automation

---

## 📌 Key Learning Outcomes

* NLP preprocessing techniques
* Sequence modeling with LSTM
* Handling imbalanced datasets
* Saving & loading ML artifacts
* Deploying ML models using Streamlit
* Professional GitHub project structuring

---

## 👨‍💻 Author

**Abhishek Shelke**

* GitHub: [https://github.com/Redskull2525](https://github.com/Redskull2525)
* LinkedIn: [https://www.linkedin.com/in/abhishek-s-b98895249](https://www.linkedin.com/in/abhishek-s-b98895249)

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and feel free to fork or contribute!
