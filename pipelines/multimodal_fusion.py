def fuse_results(results):
    return sorted(results, key=lambda r: r["score"], reverse=True)

