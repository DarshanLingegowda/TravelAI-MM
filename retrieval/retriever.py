def retrieve(query_vector, store):
    # Searches the vector store using the query embedding
    # and returns the most relevant results
    return store.search(query_vector)
