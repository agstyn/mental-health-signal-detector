# AI Mental Health Companion

> An NLP-powered web application that detects mental health signals from text using a fine-tuned **DistilBERT** transformer model — with multilingual support, explainable AI, and a real-time interactive interface.

---

## Demo

### Input Interface
![Input Interface](demo%20(1).png)

### Emotion Detection — Seeking Support
![Seeking Support](demo%20(2).png)

### Full Result View
![Full Result](demo%20(3).png)

### Emotion Detection — Positive / Recovery
![Positive Recovery](demo%20(4).png)

### Therapist Insight
![Therapist Insight](demo%20(5).png)

---

## Features

- Fine-tuned **DistilBERT** model hosted on HuggingFace (`agstyn/mental-health-distilbert`)
- **Multilingual input** — supports Tamil, Malayalam, Hindi, and 20+ languages via auto-detection and translation
- **Bilingual output** — responses shown in both the user's language and English
- **Color gradient intensity slider** — green to red emotional scale, influences advice not the core prediction
- **Tiered keyword detection** — strong words always override, mild words floor at Seeking Support to prevent wrong Positive output
- **Crisis alert system** — detects severe distress keywords and shows Indian mental health helplines
- **Explainable AI with SHAP** — visualizes which words drive predictions
- **Real-time confidence score** with progress bar
- Supportive, human-friendly therapist-style responses

---

## Emotion Labels

| Label | Description |
|---|---|
| Neutral / Advice | Stable, no emotional distress detected |
| Positive / Recovery | Optimistic or recovery-oriented language |
| Seeking Support | Mild distress, looking for help |
| Depression | Signs of sadness or emotional distress |
| Stress / Anxiety | Anxious, overwhelmed, or stressed language |

---

## How It Works

1. User types how they are feeling — in any language
2. Language is auto-detected using `langdetect`
3. Input is translated to English using `deep-translator` for model analysis
4. A tiered keyword override runs before the model:
   - Strong words like `depressed`, `hopeless` → always Depression
   - Anxiety words like `stressed`, `anxious` → always Stress / Anxiety
   - Mild words like `sad`, `exhausted` → floor at Seeking Support
   - Everything else → passed to DistilBERT model
5. Results and therapist insight are translated back to the user's language
6. Intensity slider adjusts the advice text shown at the bottom

---

## Web Application

**Example:**

Input: `"I feel extremely anxious and can't stop overthinking"`

Output:
```
Detected Signal  : Stress / Anxiety
Confidence       : 0.94
Therapist Insight: It sounds like you may be feeling anxious or stressed.
                   Try deep breathing, short breaks, or talking to someone you trust.
```

---

## Explainable AI (SHAP)

SHAP (Shapley Additive Explanations) visualizes:

- Which words most influenced the model's prediction
- How emotional language is interpreted by the model
- Transparency and trust in the AI decision-making

---

## Project Structure

```
mental-health-signal-detector/
│
├── data/
│   └── mental_health_clean.csv
│
├── models/
│   └── distilbert_mental_health/
│
├── notebooks/
│   ├── explore_01.ipynb
│   └── evaluate_02.ipynb
│
├── src/
│   └── train.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation

**Clone the repository**
```bash
git clone https://github.com/agstyn/mental-health-signal-detector.git
cd mental-health-signal-detector
```

**Create and activate a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Run the App

```bash
streamlit run app.py
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Deep Learning | PyTorch, HuggingFace Transformers |
| NLP Model | DistilBERT (fine-tuned) |
| Explainability | SHAP |
| Frontend | Streamlit |
| Translation | deep-translator, langdetect |
| Data and ML | Scikit-learn, Pandas, NumPy |

---

## Future Improvements

- [ ] Deploy on HuggingFace Spaces
- [ ] Expand dataset for better accuracy on casual and short inputs
- [ ] Add conversation-based support (multi-turn chat)
- [ ] Add mood tracking over time
- [ ] Incorporate more regional Indian language support

---

## Disclaimer

This tool is for **educational purposes only** and does not replace professional mental health advice. If you or someone you know is in crisis, please contact a qualified mental health professional.

**KIRAN Mental Health Helpline (India):** 1800-599-0019  
**AASRA Helpline:** +91 9820466627

---

## Author

**Agasthyan S**  
Data Science Student

[![GitHub](https://img.shields.io/badge/GitHub-agstyn-black?logo=github)](https://github.com/agstyn)
