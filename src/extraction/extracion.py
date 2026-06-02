from pydantic import BaseModel, Field

class ExtractedNode(BaseModel):
    name: str = Field(description='Label of the extracted entity')

class ExtractedEdge(BaseModel):
    source: str
    target: str
    label: str = Field(description='Label of the relationship')

class GraphData(BaseModel):
    nodes: list[ExtractedNode]
    edges: list[ExtractedEdge]
