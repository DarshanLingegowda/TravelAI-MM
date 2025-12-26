def process_image(image_bytes, encoder, store):
    embedding = encoder.encode(image_bytes)
    store.add(embedding, {"type": "image"})

