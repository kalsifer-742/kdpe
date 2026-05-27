import json
from pathlib import Path
import networkx as nx
import random
import ollama

from pydantic import BaseModel

#TODO from utils.email import sample_emails
#TODO filter out emails with Spacy
#TOOD cleanup graph by removing duplicated nodes

def format_emails(emails):
    formatted = []
    for email in emails:
        formatted.append(f"From: {email['from']}\nTo: {email['to']}\nSubject: {email['subject']}\n{email['body'][:500]}...") #TODO reading the entire mail crashes the program
    return "\n\n---\n\n".join(formatted)

class ExtractedNode(BaseModel):
    name: str

class ExtractedEdge(BaseModel):
    source: str
    target: str

class GraphData(BaseModel):
    nodes: list[ExtractedNode]
    edges: list[ExtractedEdge]

class Agent:
    def __init__(self, system_prompt, schema):
        self.system_prompt_content = system_prompt["extraction"]["content"].replace("{schema}", json.dumps(schema))

    def extract(self, email):
        email_string = f"From: {email.get('from', '')}\nTo: {email.get('to', '')}\nSubject: {email.get('subject', '')}\n\n{email.get('body', '')}"

        messages = [
            {"role": "system", "content": self.system_prompt_content},
            {"role": "user", "content": email_string}
        ]

        response = ollama.chat(
            model = "gemma4:e4b",
            messages = messages,
            format = GraphData.model_json_schema(),
            options = {"temperature": 0.2}
        )

        graph_update = json.loads(response['message']['content'])
        print(graph_update)
        nodes = [node["name"] for node in graph_update['nodes']]
        edges = [(edge["source"], edge["target"]) for edge in graph_update['edges']]
        return (nodes, edges)

#TODO filter out emails with Spacy
if __name__ == "__main__":
    emails = [json.loads(l) for l in open("data/splits/validation.json")]
    random.seed(742)
    sampled_emails = random.sample(emails, 3)
    schema = json.load(open("schemas/conversational.json"))
    system_prompt = json.loads(Path("prompts/system.json").read_text())

    agent = Agent(system_prompt, schema)
    graph = nx.MultiGraph()
    for email in sampled_emails:
        print(email)
        (nodes, edges) = agent.extract(email)
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

    nx.write_gexf(graph, "graphs/graph.gexf")
