class VectorStore:
    def __init__(self):
        self.vectors = []
        self.metadata = []

    def add(self, vector, meta):
        self.vectors.append(vector)
        self.metadata.append(meta)

    def search(self, query_vector):
        return []

