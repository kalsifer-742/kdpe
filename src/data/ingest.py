from pathlib import Path
import email
from email import policy
import json
import tqdm
from rich import print
from rich.table import Table

def print_summary(records, files_n):
    ingested_n = len(records)
    senders_n = sum(1 for r in records if r['from'])
    recipients_n = sum(1 for r in records if r['to'])
    non_empty_bodies_n = sum(1 for r in records if len(r['body']) >= 1)
    body_lengths = [len(r['body']) for r in records]
    min_len = min(body_lengths)
    max_len = max(body_lengths)
    avg_len = sum(body_lengths) / len(body_lengths)

    table = Table(title="Ingestion Metrics", style="cyan")
    table.add_column("Metric", justify="left", style="white")
    table.add_column("Value", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="yellow")

    table.add_row("Total Emails Ingested", str(ingested_n), f"{ingested_n/files_n*100:.1f}%")
    table.add_row("Has Sender", str(senders_n), f"{senders_n/ingested_n*100:.1f}%")
    table.add_row("Has Recipient", str(recipients_n), f"{recipients_n/ingested_n*100:.1f}%")
    table.add_row("Non-empty Bodies", str(non_empty_bodies_n), f"{non_empty_bodies_n/ingested_n*100:.1f}%")
    table.add_section()
    table.add_row("Min Body Length", f"{min_len}", "chars")
    table.add_row("Max Body Length", f"{max_len:,}", "chars")
    table.add_row("Avg Body Length", f"{avg_len:.1f}", "chars")

    print("\n")
    print(table)

def parse_eml(path):
    with open(path, "rb") as f: #open read in binary format
        msg = email.message_from_binary_file(f, policy=policy.default)
    
    body_part = msg.get_body(preferencelist=("plain", "html"))
    body_text = body_part.get_content() if body_part else ""
    
    return {
        "subject": msg["subject"],
        "from": msg["from"],
        "to": msg["to"],
        "body": body_text
    }

def ingest(files_paths, records):
    for file_path in tqdm.tqdm(files_paths, desc="Ingesting emails", unit="file"):
        try:
            record = parse_eml(file_path)
            records.append(record)
        except Exception as e:
            tqdm.write(f"Skipping file {file_path} due to error: {e}")

if __name__ == "__main__":
    source_path = Path("data/raw")
    output_path = Path("data/processed/emails.json")
    output_path.parent.mkdir(exist_ok=True)
    
    files_paths = list(source_path.iterdir())
    records = []

    ingest(files_paths, records)

    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print_summary(records, len(files_paths))