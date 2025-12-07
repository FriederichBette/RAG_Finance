# finance_rag_gpt2.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# --------------------------
# 1. Unternehmensdaten mit festen Zahlen
# --------------------------
finance_docs = [
    "AlphaCorp: Revenue $120M, Profit $15M, Growth 5.2%.",
    "BetaInc: Revenue $300M, Profit $45M, Growth 12.0%.",
    "GammaLLC: Revenue $250M, Profit $30M, Growth 8.5%.",
    "DeltaSolutions: Revenue $400M, Profit $60M, Growth 10.0%.",
    "EpsilonTech: Revenue $180M, Profit $20M, Growth 6.5%."
]

# --------------------------
# 2. FAISS Index bauen
# --------------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embed_model.encode(finance_docs)
dim = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(doc_embeddings))

# --------------------------
# 3. GPT-2 Modell laden
# --------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")

def generate_gpt2_answer(prompt, max_new_tokens=50):
    """Kurze, konservative Antwort generieren"""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,  # statt max_length
        do_sample=False,                # konservativ
        temperature=0.7,
        top_k=20,
        top_p=0.9,
        repetition_penalty=2.0,
        pad_token_id=tokenizer.eos_token_id
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    return answer

# --------------------------
# 4. RAG + GPT-2 Abfrage
# --------------------------
def ask_finance(question, top_k=3):
    # FAISS Retrieval
    query_vec = embed_model.encode([question])
    D, I = index.search(np.array(query_vec), k=top_k)
    retrieved_text = "\n".join([finance_docs[i] for i in I[0]])
    
    # GPT-2 Prompt
    prompt = f"Data:\n{retrieved_text}\nQuestion: {question}\nAnswer:"
    return generate_gpt2_answer(prompt)

# --------------------------
# 5. Interaktive Abfrage
# --------------------------
if __name__ == "__main__":
    print("Finance RAG Bot with GPT-2 is ready. Ask questions about the companies.")
    while True:
        question = input("\nAsk a finance question (or type 'exit'): ")
        if question.lower() in ["exit", "quit"]:
            break
        answer = ask_finance(question)
        print(f"\nFinanceBot says: {answer}")
