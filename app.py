import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ----------------------------
# Load tokenizer and model
# ----------------------------

MODEL_PATH = "models/distilbert_mental_health"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

# ----------------------------
# Label mapping
# ----------------------------

label_map = {
    0: "Stress / Anxiety",
    1: "Depression",
    2: "Seeking Support",
    3: "Neutral / Advice",
    4: "Positive / Recovery"
}

response_map = {
    0: "It sounds like you may be feeling stressed or anxious. Try taking a short break, slow your breathing, and talk to someone you trust.",
    
    1: "Your message shows signs of sadness or emotional distress. Remember that you are not alone. Consider reaching out to a friend, family member, or a mental health professional.",
    
    2: "It seems like you may be looking for support. Talking with others who understand your situation can really help.",
    
    3: "Your message appears neutral or informational. If you ever feel overwhelmed, stepping away for a moment and reflecting can help clear your thoughts.",
    
    4: "Your message reflects a positive or recovering mindset. Keep taking care of yourself and continue doing what helps you stay well."
}

# ----------------------------
# Streamlit UI
# ----------------------------

st.title("Mental Health Signal Detector")

st.write(
"Enter a sentence and the AI model will detect the mental health signal."
)

user_input = st.text_area(
    "Enter text",
    placeholder="Type how you are feeling..."
)

# ----------------------------
# Prediction logic
# ----------------------------

if st.button("Analyze") and user_input.strip() != "":

    inputs = tokenizer(
        user_input,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)

    prediction = torch.argmax(probs, dim=1).item()
    confidence = probs[0][prediction].item()

    label = label_map[prediction]
    message = response_map[prediction]

    # ----------------------------
    # Display results
    # ----------------------------

    st.subheader("Detected Mental Health Signal")
    st.success(label)

    st.write("Confidence:", round(confidence, 3))

    st.subheader("Supportive Suggestion")

    if prediction == 1:
        st.error(message)

    elif prediction == 0:
        st.warning(message)

    elif prediction == 2:
        st.info(message)

    else:
        st.success(message)

# ----------------------------
# Disclaimer
# ----------------------------

st.markdown("---")

st.caption(
"This tool is for educational purposes only and does not replace professional mental health advice."
)