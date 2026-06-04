from networkx import MultiDiGraph
import networkx as nx
from rich.console import Console
from sentence_transformers import SentenceTransformer
from rich.table import Table
from tqdm import tqdm

def print_summary(console: Console, initial_nodes_n, resolved_nodes_n):
    table = Table(title="Resolution Metrics", style="cyan")
    table.add_column("Metric", justify="left", style="white")
    table.add_column("Value", justify="right", style="green")
    table.add_column("Reduced", justify="right", style="yellow")

    table.add_row("Initial Nodes Number", f"{initial_nodes_n}", "")
    table.add_row("After Resolution Nodes Number", f"{resolved_nodes_n}", f"{(initial_nodes_n - resolved_nodes_n)/initial_nodes_n*100:.1f}%")

    console.print(table)

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

    connected_components = nx.connected_components(similarity_graph)
    
    for component in tqdm(connected_components, desc="resolving graph", unit="component"):        
        # heuristic: longest as canonical name
        canonical_node = max(list(component), key=len)

        for node in tqdm(component, desc="resolving component", unit="node", position=1, leave=False):
            if node != canonical_node:
                for u, v, data in graph.in_edges(node, data=True):
                    graph.add_edge(u, canonical_node, **data)
                for u, v, data in graph.out_edges(node, data=True):
                    graph.add_edge(canonical_node, v, **data)
                graph.remove_node(node)

    return graph