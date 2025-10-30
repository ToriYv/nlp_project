import os
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler,
)
from torch.optim import AdamW
from datasets import Dataset as HFDataset
from tqdm.auto import tqdm
import evaluate


# Добавляем корневую папку в sys.path (на всякий случай)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.helpers import analyze_dataset, plot_class_distribution


# ==================== КОНФИГУРАЦИЯ ====================
MODEL_NAME = "cointegrated/rubert-tiny2"
NUM_LABELS = 2
BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
FREEZE_BACKBONE = True


def main():
    print("🚀 Запуск проекта: Сентимент-анализ")

    # Загружаем данные (путь относительно корня проекта)
    df = pd.read_csv('data/sentiment_dataset.csv')
    print("📁 Загружены данные:")
    analyze_dataset(df)
    plot_class_distribution(df)

    # Преобразуем в Hugging Face Dataset
    dataset = HFDataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.3, seed=42)
    print(f"📈 Разделение: Train {len(dataset['train'])}, Test {len(dataset['test'])}")

    # ==================== ТОКЕНИЗАЦИЯ ====================
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )

    print("🔤 Токенизация данных...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=BATCH_SIZE)

    # ==================== МОДЕЛЬ ====================
    print(f"🤖 Загружаем модель {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )

    if FREEZE_BACKBONE:
        for param in model.base_model.parameters():
            param.requires_grad = False
        print("✅ Тело модели заморожено, обучается только головка!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"📱 Используется устройство: {device}")

    # ==================== ОПТИМИЗАТОР И ПЛАНИРОВЩИК ====================
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    metric = evaluate.load("accuracy")

    # ==================== ОБУЧЕНИЕ ====================
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    print("🎯 Начинаем обучение...")

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"📊 Epoch {epoch + 1} завершён. Средний loss: {avg_loss:.4f}")

    # ==================== ОЦЕНКА ====================
    print("🧪 Оцениваем модель...")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    final_score = metric.compute()
    print(f"🎉 Финальная точность: {final_score}")

    # ==================== СОХРАНЕНИЕ ====================
    model_dir = "models/sentiment_model"
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"💾 Модель сохранена в {model_dir}/")

    # ==================== ИНФЕРЕНС ====================
    def predict_sentiment(text):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][prediction].item()
        sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
        return sentiment, confidence

    test_texts = [
        "Это был хороший фильм с интересным сюжетом",
        "Ужасная картина, полный провал",
    ]

    print("\n🔍 Тестируем модель на примерах:")
    for text in test_texts:
        sentiment, confidence = predict_sentiment(text)
        print(f"📝 '{text}'")
        print(f"   → {sentiment} (уверенность: {confidence:.1%})")


if __name__ == "__main__":
    main()