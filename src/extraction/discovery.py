from pydantic import BaseModel, Field
from typing import List
from rich.table import Table

class Schema(BaseModel):
    clarifying_question: str = Field(description="If more details are needed to complete the schema, write a question for the user here.")
    entities: List[str] = Field(description='List of allowed entities, ex: ["Person", "Company", "Project"]')
    relationships: List[tuple[str, str]] = Field(description='List of allowed relationships withe their description, ex: [("flied_with", "Two people who traveled together, ex: flights or trips"), ("employed_at", "A person who is employed at an organization")]')

class Entity(BaseModel):
    raw_mention: str = Field(description="The exact text found in the email, e.g., 'the President' or 'POTUS'")
    normalized_name: str = Field(description="The actual, real-world specific name of the entity, resolving titles to specific people/organizations where possible (e.g., 'Donald Trump')")
    type: str = Field(description="PERSON, ORGANIZATION, etc.")

class Relationship(BaseModel):
    source: str = Field(description="The normalized_name of the source entity")
    target: str = Field(description="The normalized_name of the target entity")
    relationship_type: str = Field(description="The type of relationship, e.g., 'MEETS_WITH'")

def print_schema(console, entities, relationships):
    table = Table(title="Schema", style="cyan")
    table.add_column("Field", justify="left", style="white")
    table.add_column("Value", justify="left", style="green")

    table.add_row("Entities", f"{', '.join(entities)}")
    table.add_section()
    table.add_row("Relationships", "")
    for rel, desc in relationships:
        table.add_row(f"  - {rel}", desc)
    
    console.print(table)