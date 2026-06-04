from pydantic import BaseModel, Field
from tqdm import tqdm
from utils import email as email_tools
import networkx as nx

class Entity(BaseModel):
    raw_mention: str = Field(description="The exact text found in the email, e.g., 'the President' or 'POTUS'")
    normalized_label: str = Field(description="The actual, real-world specific name of the entity, resolving titles to specific people/organizations where possible (e.g., 'Donald Trump')")

class Relationship(BaseModel):
    source: str = Field(description="The normalized_label of the source entity")
    target: str = Field(description="The normalized_label of the target entity")
    label: str = Field(description='Label of the relationship')

class GraphData(BaseModel):
    nodes: list[Entity]
    edges: list[Relationship]

def extract_graph(emails, agent, prompt):
    graph = nx.MultiDiGraph()

    for email in tqdm(emails, desc="extracing emails", unit="email"):
        response = agent.chat_batch([prompt, {"role": "user", "content": email_tools.format_email(email)}])
        (nodes, edges) = extract_graph_nx(response)
        for node in nodes:
            graph.add_node(node[0], raw_mention=node[1])
        for edge in edges:
            graph.add_edge(edge[0], edge[1], label=edge[2])
    return graph

def extract_graph_nx(graph_data: GraphData):
    nodes = [(node["normalized_label"], node["raw_mention"]) for node in graph_data["nodes"]]
    edges = [(edge["source"], edge["target"], edge["label"]) for edge in graph_data["edges"]]
    return (nodes, edges)