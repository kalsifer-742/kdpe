from pathlib import Path
import random
import json
from tqdm import tqdm
from rich import print
from rich.table import Table
from rich.console import Console

SEED = 742
TEST_SIZE = 0.3
VALIDATION_SIZE = 1 - TEST_SIZE
console = Console()

def print_summary(records, validation_set, training_set):
    records_n = len(records)
    validation_set_len = len(validation_set)
    training_set_len = len(training_set)

    table = Table(title="Split Metrics", style="cyan")
    table.add_column("Info", justify="left", style="white")
    table.add_column("Value", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="yellow")

    table.add_row("Validation set", str(validation_set_len), f"{validation_set_len/records_n*100:.1f}%")
    table.add_row("Testing set", str(training_set_len), f"{training_set_len/records_n*100:.1f}%")

    console.print("\n")
    console.print(table)

def split(records):
    console.print(f"Shuffling and splitting the data with seed {SEED}")
    random.seed(SEED)
    random.shuffle(records)
    console.print("Data shuffled")

    split_index = round(len(records) * VALIDATION_SIZE)
    validation_set = records[:split_index]
    testing_set = records[split_index:]

    return validation_set, testing_set

if __name__ == "__main__":
    output_path = Path("data/splits")
    output_path.mkdir(exist_ok=True)

    with open("data/processed/emails.json", "r") as f:
            lines = f.readlines()
            records = [json.loads(line) for line in lines]

    validation_set, testing_set = split(records)

    with open(output_path / "validation.json", "w") as f:
        for record in tqdm(validation_set, desc="Writing validation set"):
            f.write(json.dumps(record) + "\n")

    with open(output_path / "test.json", "w") as f:
        for record in tqdm(testing_set, desc="Writing test set"):
            f.write(json.dumps(record) + "\n")

    print_summary(records, validation_set, testing_set)
