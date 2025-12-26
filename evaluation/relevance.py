def relevance_score(results):
    return sum(r["score"] for r in results) / max(len(results), 1)

