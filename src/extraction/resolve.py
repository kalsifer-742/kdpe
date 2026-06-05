from networkx import MultiDiGraph
import networkx as nx
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rich.table import Table
from utils import log_and_print

def print_summary(initial_nodes_n, resolved_nodes_n):
    table = Table(title="Resolution Metrics", style="cyan")
    table.add_column("Metric", justify="left", style="white")
    table.add_column("Value", justify="right", style="green")
    table.add_column("Reduced", justify="right", style="yellow")

    table.add_row("Initial Nodes Number", f"{initial_nodes_n}", "")
    table.add_row("After Resolution Nodes Number", f"{resolved_nodes_n}", f"{(initial_nodes_n - resolved_nodes_n)/initial_nodes_n*100:.1f}%")

    log_and_print(table)

def resolve_graph(graph: MultiDiGraph, model_name, threshold):
    model = SentenceTransformer(model_name)
    nodes = list(graph.nodes())
    embeddings = model.encode(nodes, convert_to_tensor=True)
    sim_matrix = model.similarity(embeddings, embeddings)

    similarity_graph = nx.Graph()
    similarity_graph.add_nodes_from(nodes)

    for i in tqdm(range(len(nodes)), desc="computing similarities", unit="node"):
        for j in range(i + 1, len(nodes)): #the matrix is symmetric
            similarity = sim_matrix[i][j].item() #converts to a simple float
            if similarity >= threshold:
                similarity_graph.add_edge(nodes[i], nodes[j])

    # possible improvement: greedy_modularity_communities()
    connected_components = list(nx.connected_components(similarity_graph))
    
    for component in tqdm(connected_components, desc="resolving graph", unit="component"):    
        if len(component) == 1: #skipping isolated nodes
            continue

        canonical_node = max(component, key=lambda n: graph.degree(n))

        for node in tqdm(component, desc="resolving component", unit="node", position=4):
            if node == canonical_node:
                continue

            in_edges = list(graph.in_edges(node, data=True))
            out_edges = list(graph.out_edges(node, data=True))

            for u, _, data in in_edges:
                if u != node: #skipping self-loops
                    graph.add_edge(u, canonical_node, **data)
            for _, v, data in out_edges:
                if v != node: #skipping self-loops
                    graph.add_edge(canonical_node, v, **data)
            graph.remove_node(node)

    return graph