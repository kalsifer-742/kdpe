from sentence_transformers import SentenceTransformer, util

def resolve_graph(G, threshold=0.88):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    nodes = list(G.nodes())
    embeddings = model.encode(nodes)

    # find pairs to merge
    merge_map = {n: n for n in nodes}
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            sim = util.cos_sim(embeddings[i], embeddings[j]).item()
            if sim > threshold:
                canonical = max(nodes[i], nodes[j], key=len)
                merge_map[nodes[i]] = canonical
                merge_map[nodes[j]] = canonical

    # apply merges
    for old, canonical in merge_map.items():
        if old == canonical or old not in G:
            continue
        # redirect all edges
        for u, v, data in list(G.in_edges(old, data=True)):
            G.add_edge(merge_map.get(u, u), canonical, **data)
        for u, v, data in list(G.out_edges(old, data=True)):
            G.add_edge(canonical, merge_map.get(v, v), **data)
        G.remove_node(old)

    return G