import json
from pathlib import Path
import random
from rich.console import Console
from rich.panel import Panel
from data import ingest
from data import process
from data import split
from tqdm import tqdm
from utils.agent import Agent
from dotenv import load_dotenv
from extraction import discovery
from extraction import resolve
from utils import email as email_tools
import extraction
import networkx as nx
import argparse
import time
import webbrowser
import os

CONFIG_PATH = Path("config.json")
RAW_DATA_DIR_PATH = Path("data/raw")
PROCESSED_MAILS_PATH = Path("data/processed/emails.jsonl")
PROCESSED_MAILS_PATH.parent.mkdir(exist_ok=True)
VALIDATION_SET_PATH = Path("data/sets/validation.json")
TEST_SET_PATH = Path("data/sets/test.json")
VALIDATION_SET_PATH.parent.mkdir(exist_ok=True)
SYSTEM_PROMPT_PATH = Path("prompts/system.json")
# USER_PROMPT_PATH = Path("prompts/user.json")
SCHEMA_PATH = Path("schemas/")
SCHEMA_PATH.mkdir(exist_ok=True)
GRAPH_PATH = Path("graphs/")
GRAPH_PATH.mkdir(exist_ok=True)
LOG_PATH = Path("logs/")
LOG_PATH.mkdir(exist_ok=True)

if __name__ == "__main__":
    load_dotenv()
    env = os.environ
    api_key = env["MISTRAL_API_KEY"]
    config = json.load(CONFIG_PATH.open("r"))
    parser = argparse.ArgumentParser(prog="kdpe - demo")
    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--skip-extraction", action="store_true")
    args = parser.parse_args()

    run_tag = f"{time.time()}_{config['seed']}"
    random.seed(config["seed"])
    console = Console(record=True)
    system_prompt = json.load(SYSTEM_PROMPT_PATH.open("r"))
    # user_prompt = json.load(USER_PROMPT_PATH.open("r"))
    agent = Agent(api_key, config["model"], temperature=config["temperature"], format=discovery.Schema)
    
    emails = []

    console.print(Panel("[magenta]CONFIGURATION"))
    console.print(config)

    console.print(Panel("[magenta]INGESTING"))

    files_paths = list(RAW_DATA_DIR_PATH.iterdir()) #[...] does not what i want
    if config["dev"]["enabled"]:
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

    if not args.skip_discovery:
        console.print(Panel("[magenta]DISCOVERY"))

        discovery_prompt = system_prompt["discovery"]
        emails_as_str = []
        for email in random.sample(validation_set, config["samples"]):
            emails_as_str.append(email_tools.format_email(email))
        discovery_prompt["content"] = discovery_prompt["content"].replace("{emails}", "\n---\n".join(emails_as_str))
        response = agent.chat(discovery_prompt)

        # Writint One-Shot Schema
        (SCHEMA_PATH / f"{run_tag}_one_shot.json").write_text(json.dumps({
            "entities": response['entities'],
            "relationships": response['relationships']
        }, indent=4))

        discovery.print_schema(console, response['entities'], response['relationships'])
        console.print(f"[bold cyan]Agent[/bold cyan] >>> {response['clarifying_question']}")

        for turn in range(config["turns"]):
            user_message = console.input("[bold cyan]User[/bold cyan] >>> ")
            response = agent.chat({"role": "user", "content": user_message})
            discovery.print_schema(console, response['entities'], response['relationships'])
            console.print(f"[bold cyan]Agent[/bold cyan] >>> {response['clarifying_question']}")

        final_schema = json.loads(agent.get_history()[-1]['content'])
        (SCHEMA_PATH / f"{run_tag}_conversational.json").write_text(json.dumps({
            "entities": final_schema['entities'],
            "relationships": final_schema['relationships']
        }, indent=4))
    
    if not args.skip_extraction:
        console.print(Panel("[magenta]EXTRACTION"))

        agent.set_format(extraction.GraphData)
        one_shot_schema = (SCHEMA_PATH / f"{run_tag}_one_shot.json").read_text()
        conversational_schema = (SCHEMA_PATH / f"{run_tag}_conversational.json").read_text()
        extraction_prompt = system_prompt["extraction"]

        if config["dev"]["enabled"]:
            emails = emails[:round(len(emails) * config["dev"]["extraction_data"])]

        console.print(Panel.fit("[cyan]ONE-SHOT"))
        extraction_prompt["content"] = extraction_prompt["content"].replace("{schema}", one_shot_schema)
        one_shot_graph = extraction.extract_graph(emails, agent, extraction_prompt)
        nx.write_gexf(one_shot_graph, GRAPH_PATH / f"{run_tag}_one_shot.gexf")

        console.print(Panel.fit("[cyan]CONVERSATIONAL"))
        extraction_prompt["content"] = extraction_prompt["content"].replace("{schema}", conversational_schema)
        conversational_graph = extraction.extract_graph(emails, agent, extraction_prompt)
        nx.write_gexf(conversational_graph, GRAPH_PATH / f"{run_tag}_conversational.gexf")

        console.print(Panel("[magenta]RESOLUTION"))

        resolved_one_shot_graph = resolve.resolve_graph(one_shot_graph, config["embedding"]["model"], config["embedding"]["similarity_threshold"])
        resolve.print_summary(console, one_shot_graph.number_of_nodes(), resolved_one_shot_graph.number_of_nodes())
        nx.write_gexf(resolved_one_shot_graph, GRAPH_PATH / f"{run_tag}_resolved_one_shot.gexf")

        resolved_conversational_graph = resolve.resolve_graph(conversational_graph, config["embedding"]["model"], config["embedding"]["similarity_threshold"])
        resolve.print_summary(console, conversational_graph.number_of_nodes(), resolved_conversational_graph.number_of_nodes())
        nx.write_gexf(resolved_conversational_graph, GRAPH_PATH / f"{run_tag}_resolved_conversational.gexf")

    console.print(Panel("[magenta]EVALUATION"))

    log = LOG_PATH / f"{run_tag}_log.html"
    console.save_html(path = log)
    webbrowser.open_new_tab(log.as_posix())
    webbrowser.open_new_tab("https://lite.gephi.org")