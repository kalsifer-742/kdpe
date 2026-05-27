import json
from pathlib import Path
import random
import ollama
from pydantic import BaseModel, Field
from typing import List, Optional
import networkx as nx
from rich.console import Console
from rich.table import Table

console = Console()

#TODO one shot version

def print_schema(entities, relationships):
    table = Table(title="Schema", style="cyan")
    table.add_column("Field", justify="left", style="white")
    table.add_column("Value", justify="left", style="green")

    table.add_row("Entities", f"{', '.join(entities)}")
    table.add_section()
    table.add_row("Relationships", "")
    for rel, desc in relationships:
        table.add_row(f"  - {rel}", desc)
    
    console.print(table)

class Schema(BaseModel):
    clarifying_question: str = Field(description="If more details are needed to complete the schema, write a question for the user here.")
    entities: List[str] = Field(description='List of allowed entities, ex: ["Person", "Company", "Project"]')
    relationships: List[tuple[str, str]] = Field(description='List of allowed relationships withe their description, ex: [("flied_with", "Two people who traveled together, ex: flights or trips"), ("employed_at", "A person who is employed at an organization")]')

def format_emails(emails):
    formatted = []
    for email in emails:
        formatted.append(f"From: {email['from']}\nTo: {email['to']}\nSubject: {email['subject']}\n{email['body'][:500]}...") #TODO reading the entire mail crashes the program
    return "\n\n---\n\n".join(formatted)

def sample_emails(emails, n, seed):
    random.seed(seed)
    sampled_emails = random.sample(emails, n)
    formatted_emails = format_emails(sampled_emails)
    return formatted_emails

class Agent:
    def __init__(self):
        self.history = []

    def get_history(self):
        return self.history

    def chat(self, message):
        self.history.append(message)
        response = ollama.chat(
            model = "gemma4:e4b", #26b is the limit on my machine
            messages = self.history,
            format = Schema.model_json_schema(),
            options = {"temperature": 0.2}
        )
        self.history.append({"role": "assistant", "content": response['message']['content']})

        schema = json.loads(response['message']['content'])
        print_schema(schema['entities'], schema['relationships'])
        
        console.print(f"[bold cyan]Agent[/bold cyan] >>> {schema['clarifying_question']}")
        

if __name__ == "__main__":
    config = json.loads(Path("configs/conversation.json").read_text())
    emails = [json.loads(l) for l in open("data/splits/validation.json")]
    schema_output_path = Path("schemas/conversational.json")
    system_prompt = json.loads(Path("prompts/system.json").read_text())
    sampled_emails = sample_emails(emails, config["email_samples"], seed=config["seed"])

    agent = Agent()

    console.print("[bold magenta]Starting Conversational Schema Extraction...[/bold magenta]")

    content = system_prompt["discovery"]["content"].replace("{emails}", sampled_emails)
    agent.chat({"role": "system", "content": content})

    for turn in range(config["turns"]):
        message = console.input("[bold cyan]User[/bold cyan] >>> ")
        agent.chat({"role": "user", "content": message})

    final_schema = json.loads(agent.get_history()[-1]['content'])
    schema_output_path.write_text(json.dumps({
        "entities": final_schema['entities'],
        "relationships": final_schema['relationships']
    }, indent=4))