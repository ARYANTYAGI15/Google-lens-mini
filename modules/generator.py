# modules/generator.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

GEN_MODEL = "google/flan-t5-small"

_tokenizer = None
_model = None

PROMPT = """You are an assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say: "Answer not available in context."
Always include citations like [image_id|chunk_id].

Context:
{context}

Question: {question}
Answer:
"""

def load_generator():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    if _model is None:
        _model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)

def generate_answer(query, retrieved, max_new_tokens=128):
    load_generator()
    context = "\n".join([f"[{r['image_id']}|{r['chunk_id']}]: {r['text']}" for r in retrieved])
    prompt = PROMPT.format(context=context, question=query)

    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = _model.generate(**inputs, max_new_tokens=max_new_tokens)
    return _tokenizer.decode(outputs[0], skip_special_tokens=True)
