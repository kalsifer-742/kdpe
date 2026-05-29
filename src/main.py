import json
from pathlib import Path
import random
from rich.console import Console
from rich.panel import Panel
from data import ingest
from data import process
from data import split
from tqdm import tqdm
from utils import Agent

DEV = True
CONFIG_PATH = Path("config.json")
RAW_DATA_DIR_PATH = Path("data/raw")
PROCESSED_MAILS_PATH = Path("data/processed/emails.jsonl")
PROCESSED_MAILS_PATH.parent.mkdir(exist_ok=True)
VALIDATION_SET_PATH = Path("data/sets/validation.json")
TEST_SET_PATH = Path("data/sets/test.json")
VALIDATION_SET_PATH.parent.mkdir(exist_ok=True)
SYSTEM_PROMPT_PATH = Path("prompts/systems.json")
USER_PROMPT_PATH = Path("prompts/user.json")

if __name__ == "__main__":
    config = json.load(CONFIG_PATH.open("r"))
    random.seed(config["seed"])
    console = Console()
    system_prompt = json.load(SYSTEM_PROMPT_PATH.open("r"))
    user_prompt = json.load(USER_PROMPT_PATH.open("r"))
    agent = Agent()
    emails = []

    console.print(f"Seed: {config["seed"]}")
    console.print(Panel("[magenta]INGESTING"))

    files_paths = list(RAW_DATA_DIR_PATH.iterdir()) #[...] does not what i want
    if DEV:
        files_paths = files_paths[:round(len(files_paths) * config["dev"]["data"])]
    for file_path in tqdm(files_paths, desc="Ingesting emails", unit="file"):
        email = ingest.parse(file_path)
        if email:
            emails.append(email)

    ingest.print_summary(console, emails, len(files_paths))

    for email in tqdm(emails, desc="Processing emails", unit="email"):
        process.process(email)
            
    with PROCESSED_MAILS_PATH.open("w") as f:
        for email in tqdm(emails, desc="Writing emails to file", unit="email"):
            f.write(json.dumps(email) + "\n")

    console.print(Panel("[magenta]SPLITTING"))

    console.print(f"Splitting the data...")
    (validation_set, test_set) = split.split(emails, config["test_size"])

    split.print_summary(emails, validation_set, test_set)

    with open(VALIDATION_SET_PATH, "w") as f:
        for email in tqdm(validation_set, desc="Writing validation set"):
            f.write(json.dumps(email) + "\n")

    with open(TEST_SET_PATH, "w") as f:
        for email in tqdm(test_set, desc="Writing test set"):
            f.write(json.dumps(email) + "\n")

    console.print(Panel("[magenta]EXTRACTION"))

    system_prompt = system_prompt["discovery"].replace("{emails}", random.sample(emails, config["samples"]))
    agent.chat(system_prompt)

    for turn in range(config["turns"]):
        user_message = console.input("[bold cyan]User[/bold cyan] >>> ")
        agent.chat({"role": "user", "content": user_message})

    # final_schema = json.loads(agent.get_history()[-1]['content'])
    # schema_output_path.write_text(json.dumps({
    #     "entities": final_schema['entities'],
    #     "relationships": final_schema['relationships']
    # }, indent=4))