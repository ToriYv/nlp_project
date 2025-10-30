import pandas as pd
import matplotlib.pyplot as plt


def analyze_dataset(df, text_col='text', label_col='label'):
    """Анализирует датасет и выводит статистику."""
    print("АНАЛИЗ ДАТАСЕТА:")
    print(f" Всего примеров: {len(df)}")
    print("Размеры классов:")

    for label in sorted(df[label_col].unique()):
        count = len(df[df[label_col] == label])
        percentage = count / len(df) * 100
        print(f" - Класс {label}: {count} примеров ({percentage:.2f}%)")

    print("Примеры текстов:")
    for i in range(min(3, len(df))):
        text_preview = (
            df[text_col].iloc[i][:50] + "..."
            if len(df[text_col].iloc[i]) > 50
            else df[text_col].iloc[i]
        )
        print(f" {i + 1}. '{text_preview}' -> {df[label_col].iloc[i]}")


def plot_class_distribution(df, label_col='label'):
    """Показывает распределение классов."""
    counts = df[label_col].value_counts().sort_index()
    plt.figure(figsize=(8, 4))
    counts.plot(kind='bar', color=['red', 'green'])
    plt.title("Распределение классов")
    plt.xlabel("Класс")
    plt.ylabel("Количество примеров")
    plt.xticks(rotation=0)
    plt.show()