import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

MODEL_PATH = "./models/spam_model"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        "Модель не найдена! Сначала обучите её с помощью:\n"
        "python scripts/project2_spam.py"
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def classify_spam(text: str) -> dict:
    if not text.strip():
        return {"NOT SPAM": 0.5, "SPAM": 0.5}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return {
        "NOT SPAM": float(probs[0]),
        "SPAM": float(probs[1])
    }

gr.Interface(
    fn=classify_spam,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Введите SMS или сообщение...",
        label="Текст сообщения"
    ),
    outputs=gr.Label(num_top_classes=2),
    title="🇷🇺 Спам-детектор (RuBERT-tiny)",
    description="Модель обучена на небольшом датасете. Для учебных целей.",
    examples=[
        ["Вы выиграли 1 000 000 рублей! Перейдите по ссылке!"],
        ["Привет! Как дела? Встречаемся сегодня в 18:00?"]
    ],
    live=False
).launch()  # ← В Colab замените на .launch(share=True)