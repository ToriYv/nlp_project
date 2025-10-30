# 📁 scripts/project2_spam.py
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler
)
from torch.optim import AdamW
from datasets import Dataset
from tqdm.auto import tqdm
import evaluate
import pandas as pd
import os
import sys

# Добавляем корень проекта в путь
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.helpers import analyze_dataset, plot_class_distribution

# ==================== КОНФИГУРАЦИЯ ====================
MODEL_NAME = "cointegrated/rubert-tiny2"
NUM_LABELS = 2
BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
FREEZE_BACKBONE = True

def main():
    print("🚀 Запуск проекта: Спам-детекция")
    
    # Загружаем данные
    df = pd.read_csv('data/spam_dataset.csv')
    print("📁 Загружены данные:")
    analyze_dataset(df)
    plot_class_distribution(df)
    
    # Разделение
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.3, seed=42)
    print(f"📈 Разделение: Train {len(dataset['train'])}, Test {len(dataset['test'])}")
    
    # Токенизация
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256
        )
    
    print("🔤 Токенизация...")
    tokenized = dataset.map(tokenize_function, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    train_dl = DataLoader(tokenized["train"], shuffle=True, batch_size=BATCH_SIZE)
    eval_dl = DataLoader(tokenized["test"], batch_size=BATCH_SIZE)
    
    # Модель
    print(f"🤖 Загружаем {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    
    if FREEZE_BACKBONE:
        for param in model.base_model.parameters():
            param.requires_grad = False
        print("✅ Тело модели заморожено")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Оптимизатор
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_steps = NUM_EPOCHS * len(train_dl)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_steps)
    
    # Метрика
    metric = evaluate.load("accuracy")
    
    # Обучение
    print("🎯 Обучение...")
    model.train()
    progress = tqdm(range(num_steps))
    
    for epoch in range(NUM_EPOCHS):
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.update(1)
            progress.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # Оценка
    print("🧪 Оценка...")
    model.eval()
    for batch in eval_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        metric.add_batch(predictions=preds, references=batch["labels"])
    
    score = metric.compute()
    print(f"🎉 Точность: {score}")
    
    # Сохранение
    os.makedirs("./models/spam_model", exist_ok=True)
    model.save_pretrained("./models/spam_model")
    tokenizer.save_pretrained("./models/spam_model")
    print("💾 Модель сохранена в ./models/spam_model/")
    
    # Тест
    def predict_spam(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        conf = probs[0][pred].item()
        return "SPAM" if pred == 1 else "NOT SPAM", conf
    
    test_texts = [
        "Вы выиграли миллион! Перейдите по ссылке!",
        "Привет, как дела? Встречаемся сегодня?"
    ]
    
    print("\n🔍 Тест:")
    for t in test_texts:
        label, conf = predict_spam(t)
        print(f"📝 '{t}' → {label} ({conf:.1%})")

if __name__ == "__main__":
    main()