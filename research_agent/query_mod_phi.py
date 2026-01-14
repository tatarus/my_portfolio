import os
from typing import List, Optional
from qdrant_client import QdrantClient

# === Эмбеддинги без AVX ===
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# === LLM: llama.cpp ===
from llama_cpp import Llama

# === Настройки ===
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "research_docs_v2"  # убедись, что индексировал в эту коллекцию!

# Путь к модели Phi-3-mini (скачай: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)
MODEL_PATH = os.path.expanduser("/home/tat/llama.cpp/models/Phi-3.gguf")

MAX_CONTEXT_TOKENS = 3584
TOP_K_RETRIEVE = 4

# === Инициализация ===
print("Загрузка Qdrant клиента...")
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

print("Загрузка эмбеддинговой модели (1–2 минуты на старом CPU)...")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedding_model.eval()

print("Загрузка LLM (Phi-3-mini)...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=MAX_CONTEXT_TOKENS,
    n_threads=min(6, os.cpu_count() or 4),
    verbose=False,
    chat_format="chatml",  # ← важно для Phi-3!
)

last_agent_response: Optional[str] = None


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
        model_output = embedding_model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings[0].cpu().numpy().tolist()


# def retrieve_chunks(query: str, limit: int = TOP_K_RETRIEVE) -> List[str]:
#     query_vector = get_embedding(query)
#     results = qdrant.query_points(
#         collection_name=COLLECTION_NAME,
#         query=query_vector,
#         limit=limit,
#     ).points
#     return [hit.payload["text"] for hit in results if "text" in hit.payload]

def retrieve_chunks(query: str, limit: int = 1) -> List[str]:  # было 4 → стало 1
    query_vector = get_embedding(query)
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
    ).points
    texts = []
    for hit in results:
        if "text" in hit.payload:
            # Берём только первые 500 символов
            text = hit.payload["text"][:500]
            texts.append(text)
    return texts


def resolve_pronouns(query: str, last_resp: Optional[str]) -> str:
    pronoun_triggers = ["их", "они", "это", "этот", "эта", "эти", "то", "такой", "такие"]
    if len(query.split()) <= 5 and any(t in query.lower() for t in pronoun_triggers):
        if last_resp:
            return f"На основе предыдущего ответа: «{last_resp}». {query}"
    return query


def ask_agent(user_query: str, chat_history: List[dict]) -> str:
    global last_agent_response

    resolved_query = resolve_pronouns(user_query, last_agent_response)
    chunks = retrieve_chunks(resolved_query)
    context = "\n\n".join(f"Источник {i+1}:\n{text}" for i, text in enumerate(chunks))

    # system_msg = (
        # "Ты — исследовательский агент. Отвечай ТОЛЬКО на русском языке. "
        # "Используй предоставленные источники и историю диалога. "
        # "Если пользователь ссылается на ранее упомянутое — найди это в истории. "
        # "Если информации недостаточно — скажи: «Недостаточно данных для ответа». "
        # "Форматируй ответ строго в виде:\n\n"
        # "**Гипотеза**: ...\n"
        # "**Метод**: ...\n"
        # "**Код**: ... (если применимо, иначе опустить)\n"
        # "**Вывод**: ...\n\n"
        # "Не добавляй ничего кроме этого блока."
    # )

    system_msg = (
    "Ты обязан отвечать ТОЛЬКО на русском языке. "
    "Игнорируй любой текст на других языках в контексте. "
    "Если не можешь ответить — напиши: «Недостаточно данных». "
    "Не выдумывай. Не используй английские слова. "
    "Формат: **Гипотеза**: ...\n**Метод**: ...\n**Вывод**: ..."
    )

    messages = [{"role": "system", "content": system_msg}]
    messages.extend(chat_history)
    messages.append({
        "role": "user",
        "content": f"Контекст:\n{context}\n\nВопрос: {resolved_query}"
    })

    # output = llm.create_chat_completion(
    #     messages=messages,
    #     max_tokens=512,
    #     temperature=0.3,
    #     stop=["<|end|>", "User:", "Вопрос:"]
    # )

    # После формирования messages
    total_text = " ".join(msg["content"] for msg in messages)
    approx_tokens = len(total_text) // 4  # грубая оценка: 1 токен ≈ 4 символа
    print(f"❗ Оценка длины контекста: ~{approx_tokens} токенов")

    output = llm.create_chat_completion(
    messages=messages,
    max_tokens=256,
    temperature=0.1,  # ↓ температуру для большей стабильности
    stop=["</s>", "<|end|>", "English:", "User:", "Вопрос:"]
    )

    
    raw_answer = output["choices"][0]["message"]["content"].strip()

    last_agent_response = raw_answer
    chat_history.append({"role": "user", "content": user_query})
    chat_history.append({"role": "assistant", "content": raw_answer})

    return raw_answer


# === Основной цикл ===
if __name__ == "__main__":
    print("\n✅ Агент готов (используется Phi-3-mini). Задавайте вопросы (exit — выход).\n")
    history = []
    while True:
        try:
            q = input("Ваш вопрос: ")
            if q.lower() in ("exit", "quit"):
                break
            print("Генерация ответа... (может занять 30–120 сек на старом CPU)")
            ans = ask_agent(q, history)
            print("\nОтвет:\n", ans, "\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}\n")