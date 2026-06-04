from pydantic import BaseModel, Field
from typing import List
from rich.table import Table
from utils import console, log_and_print

class Schema(BaseModel):
    clarifying_question: str = Field(description="If more details are needed to complete the schema, write a question for the user here.")
    entities: List[str] = Field(description='List of allowed entities, ex: ["Person", "Company", "Project"]')
    relationships: List[tuple[str, str]] = Field(description='List of allowed relationships withe their description, ex: [("flied_with", "Two people who traveled together, ex: flights or trips"), ("employed_at", "A person who is employed at an organization")]')

def print_schema(entities, relationships):
    table = Table(title="Schema", style="cyan")
    table.add_column("Field", justify="left", style="white")
    table.add_column("Value", justify="left", style="green")

    table.add_row("Entities", f"{', '.join(entities)}")
    table.add_section()
    table.add_row("Relationships", "")
    for rel, desc in relationships:
        table.add_row(f"  - {rel}", desc)
    
    log_and_print(table)