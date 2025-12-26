def process_text(text, encoder, store):
    embedding = encoder.encode(text)
    store.add(embedding, {"type": "text", "preview": text[:200]})

