import re
import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from googletrans import Translator as GoogleTranslator
from langdetect import detect as detect_language

# ----------------------------------
# Page Setup
# ----------------------------------

st.set_page_config(
    page_title="AI Mental Health Companion",
    page_icon="🧠",
    layout="centered"
)

# ----------------------------------
# Custom CSS
# ----------------------------------

st.markdown("""
<style>

/* Hide default slider number label and tick marks */
div[data-testid="stSlider"] label { display: none; }
div[data-testid="stSlider"] [data-testid="stTickBar"] { display: none; }
div[data-testid="stSlider"] span { display: none; }

/* Gradient track on the slider */
div[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(to right, #2ecc71, #f1c40f, #e74c3c) !important;
    height: 10px !important;
    border-radius: 8px !important;
}

/* Slider thumb */
div[data-testid="stSlider"] > div > div > div > div > div {
    background: white !important;
    border: 3px solid #888 !important;
    width: 22px !important;
    height: 22px !important;
    border-radius: 50% !important;
    top: -6px !important;
}

/* Bilingual block styling */
.bilingual-box {
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-size: 15px;
    line-height: 1.7;
}
.lang-primary {
    background-color: #1e3a5f;
    border-left: 4px solid #4da6ff;
    color: #e8f4ff;
}
.lang-english {
    background-color: #1a1a2e;
    border-left: 4px solid #aaa;
    color: #cccccc;
    font-size: 13px;
}
.lang-tag {
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 4px;
    opacity: 0.7;
}
.intensity-display {
    text-align: center;
    padding: 10px;
    border-radius: 10px;
    font-size: 18px;
    font-weight: bold;
    margin: 10px 0 16px 0;
}
.slider-labels {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: #aaa;
    margin-top: 2px;
    padding: 0 2px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# Load Model (cached)
# ----------------------------------

MODEL_PATH = "agstyn/mental-health-distilbert"

@st.cache_resource
def load_model():
    tok = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    mdl = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    mdl.eval()
    return tok, mdl

tokenizer, model = load_model()

# ----------------------------------
# Label & Response Maps
# ----------------------------------

label_map = {
    0: "Neutral / Advice",
    1: "Positive / Recovery",
    2: "Seeking Support",
    3: "Depression",
    4: "Stress / Anxiety"
}

response_map = {
    0: "Your message appears neutral. If you ever feel overwhelmed, stepping away for a moment and reflecting can help.",
    1: "Your message reflects a positive mindset. Continue maintaining healthy habits, sleep, and supportive relationships.",
    2: "It seems like you may be seeking support. Talking with someone you trust can help a lot.",
    3: "Your message shows signs of sadness or emotional distress. Remember you are not alone. Consider reaching out to a trusted person or mental health professional.",
    4: "It sounds like you may be feeling anxious or stressed. Try deep breathing, short breaks, or talking to someone you trust."
}

intensity_advice_map = {
    "high":     "You indicated strong emotional intensity. Talking to someone you trust or seeking professional help may help.",
    "moderate": "Your emotions seem moderately strong. Taking a break, journaling, or speaking with someone may help.",
    "low":      "Your emotional intensity appears manageable. Continue maintaining healthy habits."
}

# ----------------------------------
# Intensity color config (green → yellow → red)
# ----------------------------------

INTENSITY_LEVELS = [
    {"value": 1, "label": "Very Low",    "color": "#2ecc71", "text": "🟢 Very Low — Feeling mostly okay"},
    {"value": 2, "label": "Low",         "color": "#a8d832", "text": "🟡 Low — Slight discomfort"},
    {"value": 3, "label": "Moderate",    "color": "#f1c40f", "text": "🟠 Moderate — Noticeably affected"},
    {"value": 4, "label": "Strong",      "color": "#e67e22", "text": "🔴 Strong — Significantly affected"},
    {"value": 5, "label": "Overwhelming","color": "#e74c3c", "text": "🔴 Overwhelming — Strongly affected"},
]

# ----------------------------------
# Helpers
# ----------------------------------

_translator = GoogleTranslator()

def safe_translate(text, src, tgt):
    try:
        if not text.strip():
            return text
        result = _translator.translate(text, src=src if src != "auto" else None, dest=tgt)
        return result.text
    except Exception:
        return text

def get_user_language(text):
    """
    Detect language with confidence check.
    Short texts (under 5 words) always return 'en' — langdetect is unreliable on short inputs.
    Only marks as foreign if confidence >= 0.90.
    """
    try:
        words = text.strip().split()
        if len(words) < 5:
            return "en"
        from langdetect import detect_langs
        results = detect_langs(text)
        top = results[0]
        if top.lang == "en":
            return "en"
        if top.prob >= 0.90:
            return top.lang
        return "en"
    except Exception:
        return "en"

def bilingual_block(translated_text, english_text, lang_name="Detected Language"):
    """Show translated language first, English below — only if language is not English."""
    st.markdown(f"""
    <div class="bilingual-box lang-primary">
        <div class="lang-tag">{lang_name}</div>
        {translated_text}
    </div>
    <div class="bilingual-box lang-english">
        <div class="lang-tag">English</div>
        {english_text}
    </div>
    """, unsafe_allow_html=True)

def english_only_block(text, style="info"):
    """Used when input is already English."""
    if style == "error":
        st.error(text)
    elif style == "warning":
        st.warning(text)
    elif style == "success":
        st.success(text)
    else:
        st.info(text)

# ----------------------------------
# Crisis Detection
# ----------------------------------

CRISIS_KEYWORDS = [
    "i want to die", "kill myself", "end my life", "suicide",
    "i don't want to live", "cant live anymore",
    "want to end it", "no reason to live", "better off dead"
]

def detect_crisis(text):
    t = text.lower()
    return any(phrase in t for phrase in CRISIS_KEYWORDS)

# ----------------------------------
# Keyword Emotion Override
# ----------------------------------

# STRONG depression keywords — always override to Depression (label 3)
# Only use words that unambiguously mean clinical/severe sadness
STRONG_DEPRESSION_KEYWORDS = [
    "depressed", "depression", "worthless", "hopeless",
    "miserable", "numb", "heartbroken", "devastated",
    "grief", "grieving", "lost hope", "no hope",
    "cant go on", "no point", "feel nothing",
    "dead inside", "broken inside", "empty inside"
]

# MILD emotion words — let the MODEL decide for these
# These are everyday words that don't always mean Depression
# The slider has NO effect on these — model output is used as-is
MILD_WORDS_LET_MODEL_DECIDE = [
    "sad", "sadness", "unhappy", "lonely", "alone",
    "crying", "guilty", "guilt", "regret", "ashamed",
    "broken", "heartache", "upset", "down", "blue"
]

# Anxiety / Stress keywords — always override to label 4
ANXIETY_KEYWORDS = [
    "anxious", "anxiety", "panic", "panicking",
    "nervous", "worried", "worry", "overthinking",
    "stressed", "stress", "overwhelmed", "terrified",
    "restless", "uneasy", "scared", "fear"
]

# Seeking support keywords — always override to label 2
# Also includes mild sad words — these should at minimum be "Seeking Support",
# never "Positive/Recovery" which is what the model wrongly outputs for them
SUPPORT_KEYWORDS = [
    "need help", "need support", "struggling",
    "can't cope", "dont know what to do", "confused",
    "sad", "sadness", "unhappy", "upset", "down",
    "lonely", "alone", "crying", "heartache",
    "regret", "guilty", "guilt", "ashamed", "broken"
]

def keyword_override(text):
    """
    Slider has NO role here. Only strong, unambiguous words trigger an override.
    Mild words like 'sad', 'lonely' are intentionally left for the model to decide.
    """
    t = text.lower()

    # Strong depression words — always Depression
    for word in STRONG_DEPRESSION_KEYWORDS:
        if re.search(r'\b' + re.escape(word) + r'\b', t):
            return 3

    # Anxiety keywords
    for word in ANXIETY_KEYWORDS:
        if re.search(r'\b' + re.escape(word) + r'\b', t):
            return 4

    # Support keywords
    for word in SUPPORT_KEYWORDS:
        if re.search(r'\b' + re.escape(word) + r'\b', t):
            return 2

    # Mild words → return None so the model decides
    return None

# ----------------------------------
# LANGUAGE NAME MAP (iso → readable)
# ----------------------------------

LANG_NAMES = {
    "ta": "Tamil", "ml": "Malayalam", "hi": "Hindi",
    "te": "Telugu", "kn": "Kannada", "bn": "Bengali",
    "mr": "Marathi", "gu": "Gujarati", "pa": "Punjabi",
    "ur": "Urdu", "fr": "French", "de": "German",
    "es": "Spanish", "zh-cn": "Chinese", "ar": "Arabic",
    "ja": "Japanese", "ko": "Korean", "ru": "Russian",
    "pt": "Portuguese", "it": "Italian",
}

def lang_name(code):
    return LANG_NAMES.get(code, code.upper())

# ==================================
# UI STARTS HERE
# ==================================

st.title("🧠 AI Mental Health Companion")
st.write("This AI system analyzes emotional signals from text and provides supportive insights.")

# ----------------------------------
# User Input
# ----------------------------------

user_input = st.text_area(
    "How are you feeling today?",
    placeholder="You can type in English, Malayalam, Hindi, Tamil, or any language..."
)

# ----------------------------------
# Color Gradient Intensity Slider
# ----------------------------------

st.subheader("How strong is this feeling?")

intensity_value = st.slider(
    label="intensity_slider",
    min_value=1,
    max_value=5,
    value=1,
    step=1,
    label_visibility="collapsed"
)

# ----------------------------------
# Analyze Button
# ----------------------------------

if st.button("Analyze Emotion") and user_input.strip():

    # Show colored intensity display box only after clicking
    st.markdown("""
<div class="slider-labels">
    <span>🟢 Very Low</span>
    <span>🟡 Low</span>
    <span>🟠 Moderate</span>
    <span>🔴 Strong</span>
    <span>🔴 Overwhelming</span>
</div>
""", unsafe_allow_html=True)
    level = INTENSITY_LEVELS[intensity_value - 1]
    st.markdown(
        f'<div class="intensity-display" style="background-color:{level["color"]}22; '
        f'border: 2px solid {level["color"]}; color:{level["color"]};">'
        f'{level["text"]}</div>',
        unsafe_allow_html=True
    )

    # Detect language
    user_lang = get_user_language(user_input)
    is_foreign = user_lang != "en"
    uname = lang_name(user_lang)

    # Translate to English for analysis
    translated_text = safe_translate(user_input, "auto", "en") if is_foreign else user_input

    # ---- Crisis Detection ----
    if detect_crisis(translated_text):

        crisis_en = (
            "Your message suggests **severe emotional distress**.\n\n"
            "Please reach out for help immediately."
        )
        helplines = (
            "**KIRAN Mental Health Helpline (India)** — 1800-599-0019\n\n"
            "**AASRA Helpline** — +91 9820466627"
        )

        st.error("⚠️ Crisis Alert Detected")

        if is_foreign:
            crisis_translated = safe_translate(crisis_en, "en", user_lang)
            bilingual_block(crisis_translated, crisis_en, uname)
        else:
            st.write(crisis_en)

        st.subheader("Mental Health Helplines")
        st.write(helplines)
        st.stop()

    # ---- Keyword Override then Model ----
    override = keyword_override(translated_text)

    if override is not None:
        prediction = override
        confidence = 0.90
    else:
        inputs = tokenizer(
            translated_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()

    # NOTE: Slider does NOT change the emotion prediction.
    # It only affects the advice text shown below.

    label_en   = label_map[prediction]
    message_en = response_map[prediction]

    # ---- Detected Emotional Signal ----
    st.subheader("Detected Emotional Signal" if not is_foreign else
                 safe_translate("Detected Emotional Signal", "en", user_lang) + " / Detected Emotional Signal")

    if is_foreign:
        label_translated = safe_translate(label_en, "en", user_lang)
        bilingual_block(label_translated, label_en, uname)
    else:
        if prediction == 3:   st.error(label_en)
        elif prediction == 4: st.warning(label_en)
        elif prediction == 2: st.info(label_en)
        else:                 st.success(label_en)

    # ---- AI Confidence ----
    st.subheader("AI Confidence" if not is_foreign else
                 safe_translate("AI Confidence", "en", user_lang) + " / AI Confidence")
    st.progress(float(confidence))
    st.write(round(confidence, 3))

    # ---- Therapist Insight ----
    st.subheader("AI Therapist Insight" if not is_foreign else
                 safe_translate("AI Therapist Insight", "en", user_lang) + " / AI Therapist Insight")

    if is_foreign:
        message_translated = safe_translate(message_en, "en", user_lang)
        bilingual_block(message_translated, message_en, uname)
    else:
        st.info(message_en)

    # ---- Intensity Advice ----
    if intensity_value >= 4:
        advice_en = intensity_advice_map["high"]
    elif intensity_value == 3:
        advice_en = intensity_advice_map["moderate"]
    else:
        advice_en = intensity_advice_map["low"]

    if is_foreign:
        advice_translated = safe_translate(advice_en, "en", user_lang)
        bilingual_block(advice_translated, advice_en, uname)
    else:
        if intensity_value >= 4:   st.warning(advice_en)
        elif intensity_value == 3: st.info(advice_en)
        else:                      st.success(advice_en)

# ----------------------------------
# Disclaimer
# ----------------------------------

st.markdown("---")
st.caption(
    "This AI tool is for educational purposes only and does not replace professional mental health advice."
)