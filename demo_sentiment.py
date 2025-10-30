import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

MODEL_PATH = "./models/sentiment_model"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        "Модель не найдена! Сначала обучите её с помощью:\n"
        "python scripts/project1_sentiment.py"
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def classify_sentiment(text: str) -> dict:
    if not text.strip():
        return {"NEGATIVE": 0.5, "POSITIVE": 0.5}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return {
        "NEGATIVE": float(probs[0]),
        "POSITIVE": float(probs[1])
    }

gr.Interface(
    fn=classify_sentiment,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Введите отзыв или комментарий...",
        label="Текст"
    ),
    outputs=gr.Label(num_top_classes=2),
    title="🇷🇺 Анализ тональности (RuBERT-tiny)",
    description="Определяет, является ли текст позитивным или негативным.",
    examples=[
        ["Отличный фильм! Актёры играют великолепно."],
        ["Ужасное кино, зря потратил время."]
    ],
    live=False
).launch()  # ← В Colab замените на .launch(share=True)