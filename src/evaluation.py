"""
evaluation.py — standalone evaluation script for kdpe

Usage:
    uv run src/evaluation.py --tag <tag> --label          # export CSVs for precision labeling
    uv run src/evaluation.py --tag <tag> --recall-prep    # export emails for recall annotation
    uv run src/evaluation.py --tag <tag> --show           # compute and display all metrics

Ground truth workflow for recall:
    1. Run --recall-prep → creates labels/recall_emails.txt (readable) and labels/ground_truth.csv (blank)
    2. Read each email in recall_emails.txt
    3. For every relationship you find: add one row to ground_truth.csv
       (email_id, subject, relation, object)
    4. Run --show to compute recall

Precision labeling workflow:
    1. Run --label → creates labels/<condition>_<seen|unseen>.csv
    2. Open each CSV in a spreadsheet
    3. Read the 'evidence' column, judge if the edge is correct
    4. Fill 'correct' column: 1 = correct, 0 = wrong
    5. Run --show to compute precision, CIs, and generalization gaps

NOTE: Before running evaluation, fix baseline.extract_graph to store ids on edges:
    graph.add_edge(e1.text, e2.text, label=verb, ids=email["id"])
"""

import argparse
import csv
import json
import random
from pathlib import Path

import networkx as nx
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ── Paths (mirrored from main.py, no import to avoid circular deps) ──────────
GRAPH_PATH          = Path("graphs/")
SCHEMA_PATH         = Path("schemas/")
LABELS_DIR          = Path("labels/")
LOG_PATH            = Path("logs/")
VALIDATION_SET_PATH = Path("data/sets/validation.json")
TEST_SET_PATH       = Path("data/sets/test.json")
CONFIG_PATH         = Path("config.json")

LABELS_DIR.mkdir(exist_ok=True)

console = Console(record=True)


# ─────────────────────────────────────────────────────────────────────────────
# PRECISION
# ─────────────────────────────────────────────────────────────────────────────

def sample_edges(graph: nx.MultiDiGraph, email_ids: set, n: int, seed: int) -> list[dict]:
    """
    Sample N edges from the graph whose source email is in email_ids.

    Edge attributes used (from extraction/__init__.py):
        ids      → email id the edge was extracted from
        label    → relation type
        evidence → verbatim quote from the email
    """
    candidates = [
        (u, v, data)
        for u, v, data in graph.edges(data=True)
        if data.get("ids") in email_ids
    ]

    if not candidates:
        console.print(f"[yellow]  Warning: no edges found for the given email_ids set (graph has {graph.number_of_edges()} total edges)")
        return []

    rng = random.Random(seed)
    sampled = rng.sample(candidates, min(n, len(candidates)))

    return [
        {
            "subject":  u,
            "relation": data.get("label", ""),
            "object":   v,
            "evidence": data.get("evidence", ""),
            "email_id": data.get("ids", ""),
            "correct":  "",  # annotator fills: 1 = correct, 0 = wrong
        }
        for u, v, data in sampled
    ]


def export_precision_csv(samples: list[dict], path: Path):
    """Write sampled edges to CSV for hand-labeling."""
    fields = ["subject", "relation", "object", "evidence", "email_id", "correct"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
        csv.DictWriter(f, fieldnames=fields).writerows(samples)
    console.print(f"[green]  Exported {len(samples)} edges → {path}")
    console.print("[dim]  Fill 'correct': 1 = edge is correct, 0 = wrong[/dim]")


def load_precision_labels(path: Path) -> list[int]:
    """Read a filled precision CSV and return integer labels."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labels = [int(r["correct"]) for r in rows if r.get("correct") in ("0", "1")]
    unlabeled = sum(1 for r in rows if r.get("correct") not in ("0", "1"))
    if unlabeled:
        console.print(f"[yellow]  {path.name}: {unlabeled} rows still unlabeled, skipped.")
    return labels


def bootstrap_precision(
    labels: list[int], n_boot: int = 2000, seed: int = 42
) -> tuple[float, float, float]:
    """
    Returns (mean_precision, ci_low, ci_high) using bootstrap resampling.
    With ~40 labels expect CIs of roughly ±0.07.
    """
    arr = np.array(labels, dtype=float)
    rng = np.random.default_rng(seed)
    boot = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_boot)]
    return float(arr.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ─────────────────────────────────────────────────────────────────────────────
# RECALL
# ─────────────────────────────────────────────────────────────────────────────

def export_recall_annotation(emails: list[dict], n: int, seed: int):
    """
    Export N test emails for manual recall annotation.

    Creates two files:
      labels/recall_emails.txt  — readable, one email per block
      labels/ground_truth.csv   — blank CSV the annotator fills in
    """
    rng = random.Random(seed)
    sample = rng.sample(emails, min(n, len(emails)))

    # Readable text file for the annotator
    txt_path = LABELS_DIR / "recall_emails.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, email in enumerate(sample, 1):
            f.write(f"{'='*60}\n")
            f.write(f"EMAIL {i} | id: {email['id']}\n")
            f.write(f"{'='*60}\n")
            f.write(email.get("content", "(no content)"))
            f.write("\n\n")

    console.print(f"[green]  Exported {len(sample)} emails → {txt_path}")

    # Blank annotation CSV
    gt_path = LABELS_DIR / "ground_truth.csv"
    if gt_path.exists():
        console.print(f"[yellow]  {gt_path} already exists, not overwriting.")
    else:
        with open(gt_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["email_id", "subject", "relation", "object"])
            writer.writeheader()
            # One placeholder row per email as a reminder
            for email in sample:
                writer.writerow({"email_id": email["id"], "subject": "", "relation": "", "object": ""})
        console.print(f"[green]  Created blank annotation file → {gt_path}")

    console.print(
        "\n[bold]Instructions:[/bold]\n"
        "  1. Read each email in recall_emails.txt\n"
        "  2. For every person↔person or person↔org relationship you find,\n"
        "     add one row to ground_truth.csv\n"
        "  3. subject = who initiates, relation = what connects them, object = the other party\n"
        "  4. Delete placeholder rows (where subject is empty) when done\n"
        "  5. Re-run with --show to compute recall"
    )


def load_ground_truth(path: Path) -> list[dict]:
    """Load filled ground_truth.csv, skipping placeholder/empty rows."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    valid = [r for r in rows if r.get("subject") and r.get("relation") and r.get("object")]
    skipped = len(rows) - len(valid)
    if skipped:
        console.print(f"[dim]  ground_truth.csv: skipped {skipped} empty rows[/dim]")
    return valid


def _names_match(a: str, b: str) -> bool:
    """
    Fuzzy name match: one name is contained in the other (case-insensitive).
    Handles "Jeffrey Epstein" matching "Epstein" and vice-versa.
    """
    a, b = a.lower().strip(), b.lower().strip()
    return bool(a) and bool(b) and (a in b or b in a)


def compute_recall(
    ground_truth: list[dict], graph: nx.MultiDiGraph
) -> tuple[float, int, int]:
    """
    Recall = fraction of gold triples found in the graph.

    Matching strategy:
      - Entity names: fuzzy (one contains the other)
      - Relation:     case-insensitive exact match

    Returns (recall, n_found, n_total).
    """
    found = 0
    for gt in ground_truth:
        for u, v, data in graph.edges(data=True):
            subj_match = _names_match(gt["subject"], u)
            obj_match  = _names_match(gt["object"],  v)
            rel_match  = gt["relation"].lower().strip() == data.get("label", "").lower().strip()
            if subj_match and obj_match and rel_match:
                found += 1
                break
    total = len(ground_truth)
    return (found / total if total else 0.0), found, total


# ─────────────────────────────────────────────────────────────────────────────
# NEW TYPES  (schema coverage on unseen data)
# ─────────────────────────────────────────────────────────────────────────────

def compute_new_types(graph: nx.MultiDiGraph, schema: dict, email_ids: set) -> dict:
    """
    Measures how well the schema covers the relations found in the graph.

    Applied only to edges from email_ids (the test/unseen set) so that
    we measure generalization of the schema, not in-sample fit.

    Returns counts and lists of relation types not in the schema.
    """
    schema_relations = {r["label"].lower() for r in schema.get("relationships", [])}

    # Only look at edges from the target email set
    found_relations = {
        data.get("label", "").lower()
        for _, _, data in graph.edges(data=True)
        if data.get("ids") in email_ids and data.get("label")
    }

    new_rels  = sorted(found_relations - schema_relations - {""})
    known_rels = found_relations & schema_relations
    coverage   = len(known_rels) / len(found_relations) if found_relations else 1.0

    return {
        "schema_relation_count":  len(schema_relations),
        "found_relation_count":   len(found_relations),
        "new_relation_types":     new_rels,
        "n_new_relation_types":   len(new_rels),
        "schema_coverage_ratio":  round(coverage, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def print_precision_table(results: dict):
    table = Table(title="Precision", style="cyan")
    table.add_column("Condition",  style="white")
    table.add_column("Precision",  style="green",  justify="right")
    table.add_column("95% Confidence Interval",     style="yellow", justify="right")
    table.add_column("N",          style="dim",    justify="right")

    order = [
        "baseline_unseen",
        "one_shot_seen", "one_shot_unseen",
        "conversational_seen", "conversational_unseen",
    ]
    for key in order:
        if key not in results:
            continue
        r = results[key]
        table.add_row(
            key,
            f"{r['mean']:.3f}",
            f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]",
            str(r["n"]),
        )
    console.print(table)


def print_generalization_gaps(results: dict):
    console.print("\n[bold cyan]Generalization Gaps[/bold cyan]  (seen precision − unseen precision)")
    console.print("[dim]Positive = schema works better on emails seen during discovery (possible overfitting)[/dim]\n")
    for cond in ["one_shot", "conversational"]:
        seen_key   = f"{cond}_seen"
        unseen_key = f"{cond}_unseen"
        if seen_key in results and unseen_key in results:
            gap = results[seen_key]["mean"] - results[unseen_key]["mean"]
            color = "red" if gap > 0.1 else "green"
            console.print(f"  [{cond}]: [{color}]{gap:+.3f}[/{color}]")


def print_recall_results(recall_rows: list[dict]):
    table = Table(title="Recall", style="cyan")
    table.add_column("Condition", style="white")
    table.add_column("Recall",    style="green", justify="right")
    table.add_column("Found / Total", style="dim", justify="right")
    for r in recall_rows:
        table.add_row(r["condition"], f"{r['recall']:.3f}", f"{r['found']}/{r['total']}")
    console.print(table)


def print_new_types_results(coverage_rows: list[dict]):
    table = Table(title="Schema Coverage on Unseen Data", style="cyan")
    table.add_column("Condition",      style="white")
    table.add_column("Coverage",       style="green", justify="right")
    table.add_column("New types",      style="yellow", justify="right")
    table.add_column("New type names", style="dim")
    for r in coverage_rows:
        table.add_row(
            r["condition"],
            f"{r['schema_coverage_ratio']:.2%}",
            str(r["n_new_relation_types"]),
            ", ".join(r["new_relation_types"]) or "—",
        )
    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = json.load(CONFIG_PATH.open("r"))

    parser = argparse.ArgumentParser(prog="kdpe - evaluation")
    parser.add_argument("--tag", required=True, help="Run tag (from main.py output)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--label",       action="store_true", help="Export CSVs for precision hand-labeling")
    mode.add_argument("--recall-prep", action="store_true", help="Export emails for recall annotation")
    mode.add_argument("--show",        action="store_true", help="Compute and display all metrics")
    args = parser.parse_args()

    tag = args.tag
    console.print(Panel(f"[magenta]EVALUATION[/magenta]  |  tag: [bold]{tag}[/bold]"))

    # Load sets (needed by all modes)
    validation_set = [json.loads(l) for l in VALIDATION_SET_PATH.open()]
    test_set       = [json.loads(l) for l in TEST_SET_PATH.open()]
    validation_ids = {e["id"] for e in validation_set}
    test_ids       = {e["id"] for e in test_set}

    # ── LABEL MODE ────────────────────────────────────────────────────────────
    if args.label:
        console.print(Panel.fit("[cyan]PRECISION LABELING"))
        n = config["evaluation"]["samples"]

        for condition in ["one_shot", "conversational", "baseline"]:
            graph_file = GRAPH_PATH / f"{tag}_resolved_{condition}.gexf"
            if not graph_file.exists():
                console.print(f"[yellow]Missing graph: {graph_file}, skipping.")
                continue

            console.print(f"\n[bold]{condition}[/bold]")
            graph = nx.read_gexf(graph_file)

            if condition == "baseline":
                # Baseline has no "seen" concept (no conversation happened)
                samples = sample_edges(graph, test_ids, n=n, seed=config["seed"])
                export_precision_csv(samples, LABELS_DIR / "baseline_unseen.csv")
            else:
                for ids, suffix in [(validation_ids, "seen"), (test_ids, "unseen")]:
                    samples = sample_edges(graph, ids, n=n, seed=config["seed"])
                    export_precision_csv(samples, LABELS_DIR / f"{condition}_{suffix}.csv")

    # ── RECALL PREP MODE ──────────────────────────────────────────────────────
    elif args.recall_prep:
        console.print(Panel.fit("[cyan]RECALL ANNOTATION PREP"))
        n = config["evaluation"]["annotation_samples"]
        export_recall_annotation(test_set, n=n, seed=config["seed"])

    # ── SHOW MODE ─────────────────────────────────────────────────────────────
    elif args.show:

        # --- Precision + generalization gap ---
        console.print(Panel.fit("[cyan]PRECISION"))
        precision_results = {}
        for key in ["baseline_unseen", "one_shot_seen", "one_shot_unseen",
                    "conversational_seen", "conversational_unseen"]:
            path = LABELS_DIR / f"{key}.csv"
            if not path.exists():
                continue
            labels = load_precision_labels(path)
            if not labels:
                continue
            mean, lo, hi = bootstrap_precision(labels, seed=config["seed"])
            precision_results[key] = {"mean": mean, "ci_low": lo, "ci_high": hi, "n": len(labels)}

        if precision_results:
            print_precision_table(precision_results)
            print_generalization_gaps(precision_results)
        else:
            console.print("[yellow]No labeled precision files found. Run --label first.")

        # --- Recall ---
        console.print(Panel.fit("[cyan]RECALL"))
        gt_path = LABELS_DIR / "ground_truth.csv"
        if not gt_path.exists():
            console.print("[yellow]No ground_truth.csv found. Run --recall-prep first.")
        else:
            ground_truth = load_ground_truth(gt_path)
            if not ground_truth:
                console.print("[yellow]ground_truth.csv has no filled rows yet.")
            else:
                recall_rows = []
                for condition in ["one_shot", "conversational", "baseline"]:
                    graph_file = GRAPH_PATH / f"{tag}_resolved_{condition}.gexf"
                    if not graph_file.exists():
                        continue
                    graph = nx.read_gexf(graph_file)
                    recall, found, total = compute_recall(ground_truth, graph)
                    recall_rows.append({
                        "condition": condition,
                        "recall":    recall,
                        "found":     found,
                        "total":     total,
                    })
                if recall_rows:
                    print_recall_results(recall_rows)

        # --- New relation types on unseen data ---
        console.print(Panel.fit("[cyan]SCHEMA COVERAGE ON UNSEEN DATA"))
        coverage_rows = []
        for condition in ["one_shot", "conversational"]:
            schema_file = SCHEMA_PATH / f"{tag}_{condition}.json"
            graph_file  = GRAPH_PATH  / f"{tag}_resolved_{condition}.gexf"
            if not schema_file.exists() or not graph_file.exists():
                console.print(f"[yellow]Missing schema or graph for {condition}, skipping.")
                continue
            schema = json.loads(schema_file.read_text())
            graph  = nx.read_gexf(graph_file)
            cov    = compute_new_types(graph, schema, test_ids)
            coverage_rows.append({"condition": condition, **cov})

        if coverage_rows:
            print_new_types_results(coverage_rows)

        # --- Save all metrics to JSON ---
        all_metrics = {
            "tag":       tag,
            "precision": precision_results,
            "coverage":  coverage_rows,
        }
        out_path = LOG_PATH / f"{tag}_metrics.json"
        out_path.write_text(json.dumps(all_metrics, indent=2))
        console.print(f"\n[green]All metrics saved → {out_path}")