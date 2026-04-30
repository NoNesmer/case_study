"""Fill placeholders in README.md and poster_outline.md from results.csv + recommendation.json.

Run *after* the notebook has finished. Idempotent.
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent

results_csv = ROOT / "results.csv"
rec_json = ROOT / "recommendation.json"
readme_path = ROOT / "README.md"
poster_path = ROOT / "poster_outline.md"

if not results_csv.exists():
    raise SystemExit(f"{results_csv} not found — run the notebook first.")

df = pd.read_csv(results_csv)


def fmt_table_md(df: pd.DataFrame) -> str:
    cols = ["model", "f1", "latency_median_ms", "latency_p95_ms",
            "throughput_tok_per_s", "model_size_mb"]
    pretty_h = ["Model", "Span-F1 (95% CI)", "Latency p50 (ms)", "Latency p95 (ms)",
                "Throughput (tok/s)", "Size (MB)"]
    header = "| " + " | ".join(pretty_h) + " |"
    sep = "| " + " | ".join("---" for _ in pretty_h) + " |"
    rows_md = []
    for _, r in df.iterrows():
        f1_cell = f"{r['f1']:.3f} [{r['f1_low_95']:.3f}, {r['f1_high_95']:.3f}]"
        rows_md.append(
            f"| {r['model']} | {f1_cell} | {r['latency_median_ms']:.2f} | "
            f"{r['latency_p95_ms']:.2f} | {r['throughput_tok_per_s']:,.0f} | "
            f"{r['model_size_mb']:.2f} |"
        )
    return "\n".join([header, sep] + rows_md)


def fmt_rec(rec: dict) -> str:
    return f"**{rec['model']}** (F1={rec['f1']:.3f}, p50={rec['p50_ms']:.2f} ms)"


# Replace table placeholder
table_md = fmt_table_md(df)

readme = readme_path.read_text(encoding="utf-8")
# Idempotent: replace either the placeholder OR a previously-inserted table block.
if "<!-- TABLE_PLACEHOLDER -->" in readme:
    readme = readme.replace("<!-- TABLE_PLACEHOLDER -->", table_md)
else:
    # If the table is already there, just refresh it. Find first |Model| header onwards.
    pattern = re.compile(r"\| Model \| Span-F1.*?(?=\n\n)", re.DOTALL)
    if pattern.search(readme):
        readme = pattern.sub(table_md, readme, count=1)

# Recommendation tier replacements
if rec_json.exists():
    recs = json.loads(rec_json.read_text(encoding="utf-8"))
    by_budget = {int(r["budget_ms"]): r for r in recs}
    for budget in (5, 10, 50):
        if budget in by_budget:
            readme = readme.replace(f"<!-- REC_{budget}ms -->", fmt_rec(by_budget[budget]))

readme_path.write_text(readme, encoding="utf-8")
print(f"Updated {readme_path}")

# Poster outline: also fill the headline table and recommendation
poster = poster_path.read_text(encoding="utf-8")
# Replace the manual table skeleton
poster_table_pattern = re.compile(
    r"\| Model \| Span-F1 \(95% CI\) \|.*?\| DistilBERT-NER \|.*?\|",
    re.DOTALL,
)
if poster_table_pattern.search(poster):
    poster = poster_table_pattern.sub(table_md, poster, count=1)

# Replace recommendation rows in poster ("… (filled by notebook)")
if rec_json.exists():
    for budget, label in [(5, "5 ms"), (10, "10 ms"), (50, "50 ms")]:
        if budget in by_budget:
            rec = by_budget[budget]
            # Look for "**5 ms** | ... | … (filled by notebook)"
            m = re.search(
                r"(\*\*" + label + r"\*\* \| [^|]+ \| )… \(filled by notebook\)",
                poster,
            )
            if m:
                poster = poster.replace(
                    m.group(0),
                    m.group(1) + fmt_rec(rec),
                )

poster_path.write_text(poster, encoding="utf-8")
print(f"Updated {poster_path}")
