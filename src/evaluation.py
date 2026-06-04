import json
from rich.console import console
from rich.panel import Panel
import networkx as nx

from main import GRAPH_PATH, LOG_PATH, TEST_SET_PATH, VALIDATION_SET_PATH

CONFIG_PATH = Path("config.json")
LABELS_DIR = Path("labels/")
LABELS_DIR.mkdir(exist_ok=True)

def sample_edges(graph: nx.MultiDiGraph, email_ids: set, n: int, seed: int) -> list[dict]:
    """Sample N edges from the graph whose source email is in email_ids."""
    candidate_edges = [
        (u, v, data)
        for u, v, data in graph.edges(data=True)
        if data.get("email_id") in email_ids
    ]
    rng = random.Random(seed)
    sampled = rng.sample(candidate_edges, min(n, len(candidate_edges)))
    return [
        {
            "subject": u,
            "relation": data.get("relation", ""),
            "object": v,
            "evidence": data.get("evidence", ""),
            "email_id": data.get("email_id", ""),
            "label": ""   # you fill this in
        }
        for u, v, data in sampled
    ]

def export_for_labeling(samples: list[dict], out_path: Path):
    """Write samples to CSV for hand-labeling."""
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "relation", "object", "evidence", "email_id", "label"])
        writer.writeheader()
        writer.writerows(samples)
    print(f"Saved {len(samples)} edges to {out_path} — open and fill the 'label' column (1=correct, 0=wrong)")


def bootstrap_precision(labels: list[int], n: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Returns (mean_precision, ci_low, ci_high)."""
    labels = np.array(labels)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        sample = labels[rng.integers(0, len(labels), size=len(labels))]
        scores.append(sample.mean())
    return float(labels.mean()), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def compute_metrics(labels_dir: Path) -> dict:
    """
    Reads 4 labeled CSVs and computes precision + CIs + generalization gaps.
    Expected files: one_shot_in.csv, one_shot_out.csv, conv_in.csv, conv_out.csv
    """
    results = {}
    for condition in ["one_shot_in", "one_shot_out", "conv_in", "conv_out"]:
        path = labels_dir / f"{condition}.csv"
        if not path.exists():
            print(f"Missing: {path}")
            continue
        with open(path) as f:
            labels = [int(row["label"]) for row in csv.DictReader(f) if row["label"] in ("0", "1")]
        if not labels:
            print(f"No labels found in {path}")
            continue
        mean, lo, hi = bootstrap_precision(labels)
        results[condition] = {"mean": mean, "ci_low": lo, "ci_high": hi, "n": len(labels)}

    if all(k in results for k in ["conv_in", "conv_out", "one_shot_in", "one_shot_out"]):
        results["gap_conv"]     = results["conv_in"]["mean"]     - results["conv_out"]["mean"]
        results["gap_one_shot"] = results["one_shot_in"]["mean"] - results["one_shot_out"]["mean"]

    return results


def print_metrics(console, results: dict):
    from rich.table import Table
    table = Table(title="Evaluation Results")
    table.add_column("Condition")
    table.add_column("Precision")
    table.add_column("95% CI")
    table.add_column("N")

    for condition in ["one_shot_in", "one_shot_out", "conv_in", "conv_out"]:
        if condition not in results:
            continue
        r = results[condition]
        table.add_row(
            condition,
            f"{r['mean']:.3f}",
            f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]",
            str(r["n"])
        )
    console.print(table)

    if "gap_conv" in results:
        console.print(f"\nGeneralization gap (conversational): {results['gap_conv']:+.3f}")
        console.print(f"Generalization gap (one-shot):        {results['gap_one_shot']:+.3f}")


if __name__ == "__main__":
    config = json.load(CONFIG_PATH.open("r"))
    parser = argparse.ArgumentParser(prog="kdpe - evaluation")
    parser.add_argument("--tag", action="store")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--label", action="store_true")
    args = parser.parse_args()

    if args.tag == None:
        console.print("[yellow]Error: a tag is required")
        exit(1)

    if args.show and args.label:
        console.print("[yellow]Error: --show and --label flags cannot be used together")
        exit(1)

    console.print(Panel("[magenta]EVALUATION"))

    if args.label:
        #TODO simplify
        validation_set = [json.loads(line) for line in VALIDATION_SET_PATH.open("r").readlines()]
        test_set       = [json.loads(line) for line in TEST_SET_PATH.open("r").readlines()]
        resolved_one_shot_graph = nx.read_gexf(GRAPH_PATH / f"{run_tag}_one_shot.gexf")
        resolved_conversational_graph = nx.read_gexf(GRAPH_PATH / f"{run_tag}_conversational.gexf")

        validation_ids = {e["id"] for e in validation_set}
        test_ids       = {e["id"] for e in test_set}

        n_samples = config["eval_samples"]

        prefix = "one_shot"
        for split_ids, suffix in [(validation_ids, "seen"), (test_ids, "unseen")]:
            label_path = LABELS_DIR / f"{prefix}_{suffix}.csv"
            samples = sample_edges(graph, split_ids, n=n_samples, seed=config["seed"])
            export_for_labeling(samples, label_path)

        prefix = "conversational"
        for split_ids, suffix in [(validation_ids, "seen"), (test_ids, "seen")]:
            label_path = LABELS_DIR / f"{prefix}_{suffix}.csv"
            samples = sample_edges(graph, split_ids, n=n_samples, seed=config["seed"])
            export_for_labeling(samples, label_path)

    if args.show:
        results = compute_metrics(LABELS_DIR)
        if results:
            print_metrics(console, results)
            (LOG_PATH / f"{run_tag}_metrics.json").write_text(json.dumps(results, indent=2))
        else:
            console.print("[yellow]Label files exported to labels/. Fill them in and re-run to see metrics.")