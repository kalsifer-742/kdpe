import json

from pydantic import BaseModel, Field
from tqdm import tqdm
from utils import email as email_tools
import networkx as nx

from utils import debug

class Entity(BaseModel):
    normalized_label: str = Field(description="The actual, real-world specific name of the entity, resolving titles to specific people/organizations where possible (e.g., 'Donald Trump')")
    evidence: str = Field(description="The exact text found in the email that triggered this entity to be extracted")

class Relationship(BaseModel):
    source: str = Field(description="The normalized_label of the source entity")
    target: str = Field(description="The normalized_label of the target entity")
    label: str = Field(description='Label of the relationship')
    evidence: str = Field(description="The exact text found in the email that triggered this relationship to be extracted")

class GraphData(BaseModel):
    nodes: list[Entity]
    edges: list[Relationship]

def extract_graph(emails, agent, prompt):
    graph = nx.MultiDiGraph()

    for email in tqdm(emails, desc="extracing emails", unit="email"):
        response = agent.chat_batch([prompt, {"role": "user", "content": email_tools.format_email(email)}])
        (nodes, edges) = extract_graph_nx(response)
        for node_label, evidence in nodes:
            if graph.has_node(node_label):
                graph.nodes[node_label].setdefault("evidence", []).append(evidence)
                graph.nodes[node_label].setdefault("ids", []).append(email["id"])
            else:    
                graph.add_node(node_label, evidence=[evidence], ids=[email["id"]])
        for source, target, label, evidence in edges: #nodes are automaticcaly added when adding edges. maybe code can be simplified
            graph.add_edge(source, target, label=label, evidence=evidence, ids=email["id"])
    
    return graph

def extract_graph_nx(graph_data: GraphData):
    nodes = [(node["normalized_label"], node["evidence"]) for node in graph_data["nodes"]]
    edges = [(edge["source"], edge["target"], edge["label"], edge["evidence"]) for edge in graph_data["edges"]]
    return (nodes, edges)

# DISCLAIMER: AI Generated
def sanitize_for_gexf(G: nx.Graph) -> nx.Graph:
    """
    Return a copy of G with node/edge attributes serialized to GEXF-friendly values.

    Lists, tuples and dicts are JSON-serialized so the NetworkX GEXF writer
    doesn't mistake them for dynamic attribute tuples.
    """
    new_g = G.__class__()

    # copy graph-level attributes (serialize complex types)
    for k, v in G.graph.items():
        if isinstance(v, (list, tuple, dict)):
            new_g.graph[k] = json.dumps(v)
        else:
            new_g.graph[k] = v

    # copy nodes
    for n, data in G.nodes(data=True):
        attrs = {}
        for k, v in data.items():
            if isinstance(v, (list, tuple, dict)):
                attrs[k] = json.dumps(v)
            else:
                attrs[k] = v
        new_g.add_node(n, **attrs)

    # copy edges (preserve parallel edges)
    if G.is_multigraph():
        for u, v, key, data in G.edges(keys=True, data=True):
            attrs = {}
            for k, val in data.items():
                if isinstance(val, (list, tuple, dict)):
                    attrs[k] = json.dumps(val)
                else:
                    attrs[k] = val
            new_g.add_edge(u, v, key=key, **attrs)
    else:
        for u, v, data in G.edges(data=True):
            attrs = {}
            for k, val in data.items():
                if isinstance(val, (list, tuple, dict)):
                    attrs[k] = json.dumps(val)
                else:
                    attrs[k] = val
            new_g.add_edge(u, v, **attrs)

    return new_g