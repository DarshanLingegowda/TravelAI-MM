def process_audio(audio_bytes, encoder, store):
    embedding = encoder.encode(audio_bytes)
    store.add(embedding, {"type": "audio"})

