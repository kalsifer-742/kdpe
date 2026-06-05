import json
from pathlib import Path
import random
import rich
from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm
from data import ingest
from data import process
from data import split
from utils.agent import Agent
from dotenv import load_dotenv
from extraction import baseline, discovery
from extraction import resolve
from utils import create_tag, email as email_tools, log_and_input, log_and_print, save_log
import extraction
import networkx as nx
import argparse
import webbrowser
import os
import spacy

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
    parser.add_argument("--tag", action="store")
    args = parser.parse_args()

    if args.tag:
        run_tag = args.tag
    else:
        run_tag = create_tag()
        
    random.seed(config["seed"])
    nlp = spacy.load(config["baseline"]["model"])
    system_prompt = json.load(SYSTEM_PROMPT_PATH.open("r"))
    # user_prompt = json.load(USER_PROMPT_PATH.open("r"))
    agent = Agent(api_key, config["model"], temperature=config["temperature"], format=discovery.Schema)
    
    emails = []

    log_and_print(Panel(f"[magenta]TAG[/magenta] : {run_tag}"))

    log_and_print(Panel("[magenta]CONFIGURATION"))
    log_and_print(config)

    log_and_print(Panel("[magenta]INGESTING"))

    files_paths = list(RAW_DATA_DIR_PATH.iterdir()) #[...] does not what i want
    if config["dev"]["enabled"]:
        files_paths = random.sample(files_paths, round(len(files_paths) * config["dev"]["data"]))
    for file_path in tqdm(files_paths, desc="Ingesting emails", unit="email"):
        email = ingest.parse(file_path)
        if email:
            emails.append(email)

    ingest.print_summary(emails, len(files_paths))

    for email in tqdm(emails, desc="Processing emails", unit="email"):
        process.process(email)
            
    with PROCESSED_MAILS_PATH.open("w") as f:
        for email in tqdm(emails, desc="Writing emails to file", unit="email"):
            f.write(json.dumps(email) + "\n")

    log_and_print(Panel("[magenta]SPLITTING"))

    log_and_print(f"Splitting the data...")
    (validation_set, test_set) = split.split(emails, config["test_size"])

    split.print_summary(emails, validation_set, test_set)

    with open(VALIDATION_SET_PATH, "w") as f:
        for email in tqdm(validation_set, desc="Writing validation set", unit="email"):
            f.write(json.dumps(email) + "\n")

    with open(TEST_SET_PATH, "w") as f:
        for email in tqdm(test_set, desc="Writing test set", unit="email"):
            f.write(json.dumps(email) + "\n")

    if not args.skip_discovery:
        log_and_print(Panel("[magenta]DISCOVERY"))

        discovery_prompt = system_prompt["discovery"]
        emails_as_str = []
        for email in random.sample(validation_set, round(len(validation_set) * config["conversation_samples"])):
            emails_as_str.append(email_tools.format_email(email))
        discovery_prompt["content"] = discovery_prompt["content"].replace("{emails}", "\n---\n".join(emails_as_str))
        response = agent.chat(discovery_prompt)

        # Writint One-Shot Schema
        (SCHEMA_PATH / f"{run_tag}_one_shot.json").write_text(json.dumps({
            "entities": response['entities'],
            "relationships": response['relationships']
        }, indent=4))

        discovery.print_schema(response['entities'], response['relationships'])
        log_and_print(f"[bold cyan]Agent[/bold cyan] >>> {response['clarifying_question']}")

        for turn in range(config["turns"]):
            user_message = log_and_input("[bold cyan]User[/bold cyan] >>> ")
            response = agent.chat({"role": "user", "content": user_message})
            discovery.print_schema(response['entities'], response['relationships'])
            if turn == (config["turns"] - 1):
                continue #skip printing last turn
            log_and_print(f"[bold cyan]Agent[/bold cyan] >>> {response['clarifying_question']}")

        final_schema = json.loads(agent.get_history()[-1]['content'])
        (SCHEMA_PATH / f"{run_tag}_conversational.json").write_text(json.dumps({
            "entities": final_schema['entities'],
            "relationships": final_schema['relationships']
        }, indent=4))
    
    if not args.skip_extraction:
        log_and_print(Panel("[magenta]EXTRACTION"))

        agent.set_format(extraction.GraphData)
        one_shot_schema = (SCHEMA_PATH / f"{run_tag}_one_shot.json").read_text()
        conversational_schema = (SCHEMA_PATH / f"{run_tag}_conversational.json").read_text()
        extraction_template = system_prompt["extraction"]
        prompt = extraction_template

        if config["dev"]["enabled"]:
            emails = random.sample(emails, round(len(emails) * config["dev"]["extraction_data"]))

        log_and_print(Panel.fit("[cyan]BASELINE"))
        baseline_graph = baseline.extract_graph(nlp, emails)
        nx.write_gexf(extraction.sanitize_for_gexf(baseline_graph), GRAPH_PATH / f"{run_tag}_baseline.gexf")

        log_and_print(Panel.fit("[cyan]ONE-SHOT"))
        prompt["content"] = extraction_template["content"].replace("{schema}", one_shot_schema)
        one_shot_graph = extraction.extract_graph(emails, agent, prompt)
        nx.write_gexf(extraction.sanitize_for_gexf(one_shot_graph), GRAPH_PATH / f"{run_tag}_one_shot.gexf")

        log_and_print(Panel.fit("[cyan]CONVERSATIONAL"))
        prompt["content"] = extraction_template["content"].replace("{schema}", conversational_schema)
        conversational_graph = extraction.extract_graph(emails, agent, prompt)
        nx.write_gexf(extraction.sanitize_for_gexf(conversational_graph), GRAPH_PATH / f"{run_tag}_conversational.gexf")

        log_and_print(Panel("[magenta]RESOLUTION"))

        resolved_baseline_graph = resolve.resolve_graph(baseline_graph, config["embedding"]["model"], config["embedding"]["similarity_threshold"])
        nx.write_gexf(extraction.sanitize_for_gexf(resolved_baseline_graph), GRAPH_PATH / f"{run_tag}_resolved_baseline.gexf")

        resolved_one_shot_graph = resolve.resolve_graph(one_shot_graph, config["embedding"]["model"], config["embedding"]["similarity_threshold"])
        resolve.print_summary(one_shot_graph.number_of_nodes(), resolved_one_shot_graph.number_of_nodes())
        nx.write_gexf(extraction.sanitize_for_gexf(resolved_one_shot_graph), GRAPH_PATH / f"{run_tag}_resolved_one_shot.gexf")

        resolved_conversational_graph = resolve.resolve_graph(conversational_graph, config["embedding"]["model"], config["embedding"]["similarity_threshold"])
        resolve.print_summary(conversational_graph.number_of_nodes(), resolved_conversational_graph.number_of_nodes())
        nx.write_gexf(extraction.sanitize_for_gexf(resolved_conversational_graph), GRAPH_PATH / f"{run_tag}_resolved_conversational.gexf")

    log_path = LOG_PATH / f"{run_tag}_log.html"
    save_log(log_path)
    webbrowser.open_new_tab(log_path.as_posix())
    webbrowser.open_new_tab("https://lite.gephi.org")