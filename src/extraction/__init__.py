from pydantic import BaseModel, Field

class ExtractedNode(BaseModel):
    label: str = Field(description='Label of the extracted entity')

class ExtractedEdge(BaseModel):
    source: str
    target: str
    label: str = Field(description='Label of the relationship')

class GraphData(BaseModel):
    nodes: list[ExtractedNode]
    edges: list[ExtractedEdge]

def extract_graph_nx(graph_data: GraphData):
    nodes = [node["label"] for node in graph_data["nodes"]]
    edges = [(edge["source"], edge["target"], edge["label"]) for edge in graph_data["edges"]]
    return (nodes, edges)