import os
import sys
from pathlib import Path
from typing import List, Dict

# === Эмбеддинги без AVX ===
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# === Qdrant ===
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# === PDF ===
from PyPDF2 import PdfReader

# === Настройки ===
FILE='Alfa_Equity_Rus_2024_10_2988.pdf'
SOURCES_DIR = "./sources"  # ← твоя папка
COLLECTION_NAME = "research_Alfa_Equity"  # новая коллекция
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# === Загрузка эмбеддинговой модели ===
print("Загрузка эмбеддинговой модели...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding(text: str) -> List[float]:
    if not text.strip():
        return [0.0] * 384
    encoded_input = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings[0].cpu().numpy().tolist()

def extract_text_from_file(file_path: Path) -> str:
    """Извлекает текст из .txt, .md, .pdf"""
    try:
        if file_path.suffix.lower() == ".pdf":
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        elif file_path.suffix.lower() in (".txt", ".md"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        else:
            print(f"Пропущен неподдерживаемый файл: {file_path}")
            return ""
    except Exception as e:
        print(f"Ошибка при обработке {file_path}: {e}")
        return ""

def main():
    sources = Path(SOURCES_DIR)
    if not sources.exists():
        print(f"Папка не найдена: {SOURCES_DIR}")
        sys.exit(1)

    # Подключение к Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Создание новой коллекции (384-dim)
    print(f"Создание коллекции '{COLLECTION_NAME}'...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    # Сбор всех файлов

    # files = list(sources.rglob("*"))
    # text_files = [f for f in files if f.is_file() and f.suffix.lower() in (".txt", ".md", ".pdf")]
    # print(f"Найдено {len(text_files)} документов для индексации.")
    
    text_files = []
    for pattern in ["Alfa_Equity_Rus_2024_10_2988.pdf"]:  # добавьте другие файлы при необходимости
        found = list(sources.rglob(pattern))
        text_files.extend([f for f in found if f.is_file() and f.suffix.lower() in (".txt", ".md", ".pdf")])

    print(f"Найдено {len(text_files)} документов для индексации:")
    for f in text_files:
        print(f"  - {f.relative_to(sources)}")
    

    # Индексация
    points = []
    for i, file_path in enumerate(text_files):
        print(f"[{i+1}/{len(text_files)}] Обработка: {file_path.name}")
        text = extract_text_from_file(file_path)
        if not text.strip():
            continue

        # Опционально: разбивка на чанки (здесь — целиком)
        embedding = get_embedding(text)
        payload = {
            "text": text[:10000],  # ограничим длину, если очень большой
            "source": str(file_path.relative_to(sources)),
            "file_name": file_path.name
        }
        points.append(PointStruct(id=i, vector=embedding, payload=payload))

        # Пакетная отправка каждые 20 документов (экономия памяти)
        if len(points) >= 20:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            points = []

    # Отправка остатка
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"\n✅ Успешно проиндексировано {len(text_files)} документов в коллекцию '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main()