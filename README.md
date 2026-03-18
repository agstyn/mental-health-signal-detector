# 🧠 Mental Health Signal Detector

AI-powered NLP model that detects **mental health signals from text** using a fine-tuned **DistilBERT transformer model**.

The project also includes **Explainable AI (SHAP)** to understand why the model makes predictions and an **interactive Streamlit web app**.

---

# 🚀 Features

* Transformer-based mental health text classification
* Fine-tuned **DistilBERT model**
* **Explainable AI with SHAP**
* Interactive **Streamlit web application**
* Real-time prediction with confidence score
* Human-friendly supportive responses

---

# 🧠 Model

The model is trained on mental health related text data and predicts the mental state signal.

Example signals detected:

* Anxiety
* Depression
* Stress
* Emotional distress
* Neutral / stable

---

# 🖥 Web Application

The Streamlit interface allows users to input text and receive:

* Predicted mental health signal
* Model confidence score
* Helpful supportive message

Example:

Input:

"I feel extremely anxious and stressed lately"

Output:

Prediction: Anxiety
Confidence: 0.88

Suggested message:

> It looks like you're going through stress or anxiety.
> Taking a short break, talking with someone you trust, or seeking support can help.

---

# 🔍 Explainable AI

The project uses **SHAP (Shapley Additive Explanations)** to visualize:

* Which words influence the prediction
* How the model interprets emotional language

This improves transparency and trust in the model.

---

# 📁 Project Structure

```
mental-health-signal-detector
│
├── data
│   └── mental_health_clean.csv
│
├── models
│   └── distilbert_mental_health
│
├── notebooks
│   ├── explore_01.ipynb
│   └── evaluate_02.ipynb
│
├── src
│   └── train.py
│
├── app.py
└── .gitignore
```

---

# ⚙️ Installation

Clone the repository:

```
git clone https://github.com/agstyn/mental-health-signal-detector.git
```

Create virtual environment:

```
python -m venv venv
```

Activate:

Windows

```
venv\Scripts\activate
```

Install dependencies

```
pip install -r requirements.txt
```

---

# ▶️ Run the App

```
streamlit run app.py
```

---

# 🧰 Tech Stack

* Python
* PyTorch
* HuggingFace Transformers
* SHAP
* Streamlit
* Scikit-learn
* Pandas
* NumPy

---

# 🎯 Future Improvements

* Deploy on HuggingFace Spaces
* Add mental health resource suggestions
* Add conversation-based support system
* Expand dataset for better accuracy

---

# 👨‍💻 Author

Agasthyan S
Data Science Student
