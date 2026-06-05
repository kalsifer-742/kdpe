import hashlib
from pathlib import Path
import email
from email import policy
from rich.table import Table
from utils import console, log_and_print

def print_summary(emails, files_n):
    ingested_n = len(emails)
    non_empty_bodies_n = sum(1 for eml in emails if len(eml["content"]) >= 1)
    body_lengths = [len(eml["content"]) for eml in emails]
    min_len = min(body_lengths)
    max_len = max(body_lengths)
    avg_len = sum(body_lengths) / len(body_lengths)

    table = Table(title="Ingestion Metrics", style="cyan")
    table.add_column("Metric", justify="left", style="white")
    table.add_column("Value", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="yellow")

    table.add_row("Emails Successfully Ingested", str(ingested_n), f"{ingested_n/files_n*100:.1f}%")
    table.add_row("Non-empty Bodies", str(non_empty_bodies_n), f"{non_empty_bodies_n/ingested_n*100:.1f}%")
    table.add_section()
    table.add_row("Min Body Length", f"{min_len}", "chars")
    table.add_row("Max Body Length", f"{max_len:,}", "chars")
    table.add_row("Avg Body Length", f"{avg_len:.1f}", "chars")

    log_and_print(table)

def parse_eml(path: Path):
    with open(path, "rb") as f: #open read in binary format
        email_object = email.message_from_binary_file(f, policy=policy.default)
    
    email_body = email_object.get_body(preferencelist=("plain", "html"))
    email_content = email_body.get_content()
    
    return {
        "id": path.name[:12], # id previously(during download) set at start of the file name
        "subject": email_object["subject"],
        "from": email_object["from"],
        "to": email_object["to"],
        "content": email_content
    }

def parse(file_path: Path) -> object | None:
    if file_path.suffix == ".eml":
        return parse_eml(file_path)
    else:
        return None