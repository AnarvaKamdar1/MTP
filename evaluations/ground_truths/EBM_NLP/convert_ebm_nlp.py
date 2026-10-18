"""Convert EBM-NLP (Nye et al. 2018) aggregated starting_spans annotations
into a JSONL that run.py can consume, with gold in the same PAR/INT/OUT
schema as the previous test.pkl-derived dataset.

Reads:
    documents/<PMID>.txt        raw abstract text
    documents/<PMID>.tokens     one token per line, aligned with .ann files
    annotations/aggregated/starting_spans/{participants,interventions,outcomes}/<SPLIT>/<PMID>.AGGREGATED.ann
        binary 0/1 per token (1 = inside a P/I/O span)
    annotations/aggregated/hierarchical_labels/interventions/<SPLIT>/<PMID>.AGGREGATED.ann
        finer per-token code; label 7 = "Control" (closest thing to a
        real Comparator signal in this corpus) — pulled out as a bonus
        field, not part of the core PAR/INT/OUT schema.

Output schema per line (matches the previous PAR/INT/OUT dataset):
    {
        "id": "<PMID>",
        "abstract": "<raw text>",
        "raw_entities": {
            "PAR": [span strings...],
            "INT": [span strings...],   # intervention + comparator merged,
                                         # exactly like EBM-NLP's own convention
            "OUT": [span strings...]
        },
        "comparator_hierarchical": [span strings...]  # bonus, subset of INT
                                                        # where hierarchical
                                                        # label == 7 (Control)
    }
"""

import glob
import json
import os
import tarfile

# Folder the .tar.gz gets extracted into (this is what DATA_DIR points at
# below — matches the top-level folder name inside the archive).
DATA_DIR = "ebm_nlp_2_00"

# "test/gold" = expert-annotated held-out set, the right choice for
# zero-shot evaluation. Change to "train" to convert the training split
# instead (4,792 docs, crowd-aggregated, noisier).
SPLIT = "test"

OUTPUT_FILE = "EBM_NLP_ground_truth.jsonl"


def ensure_extracted():
    """Extract the .tar.gz in this folder into DATA_DIR, if not already done."""

    if os.path.isdir(DATA_DIR):
        print(f"'{DATA_DIR}' already exists, skipping extraction.")
        return

    candidates = sorted(
        f for f in os.listdir(".")
        if f.endswith(".tar.gz") or f.endswith("_tar.gz") or f.endswith(".tgz")
    )

    if not candidates:
        raise FileNotFoundError(
            "No .tar.gz file found in the current folder, and "
            f"'{DATA_DIR}' does not exist either. Place the EBM-NLP "
            ".tar.gz next to this script, or run it from the folder "
            "that contains it."
        )

    if len(candidates) > 1:
        print(f"Multiple .tar.gz files found, using the first one: {candidates[0]}")

    archive_path = candidates[0]

    print(f"Extracting {archive_path} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(".")

    if not os.path.isdir(DATA_DIR):
        raise RuntimeError(
            f"Extraction finished but expected folder '{DATA_DIR}' was not "
            "found. Check the archive's top-level folder name and update "
            "DATA_DIR at the top of this script to match."
        )

    print(f"Extracted into '{DATA_DIR}'.")

ELEMENT_TO_LETTER = {
    "participants": "PAR",
    "interventions": "INT",
    "outcomes": "OUT",
}


def load_tokens(pmid):
    path = os.path.join(DATA_DIR, "documents", f"{pmid}.tokens")
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip() != "" or line == "\n"]


def load_labels(element, split, pmid, phase="starting_spans"):
    path = os.path.join(
        DATA_DIR, "annotations", "aggregated", phase, element, split,
        f"{pmid}.AGGREGATED.ann"
    )
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() != ""]


def spans_from_binary_labels(tokens, labels):
    """Group contiguous '1' tokens into phrase strings."""
    spans = []
    current = []

    for tok, lab in zip(tokens, labels):
        if lab == "1":
            current.append(tok)
        else:
            if current:
                spans.append(" ".join(current))
                current = []

    if current:
        spans.append(" ".join(current))

    return spans


def spans_from_code_labels(tokens, labels, target_code):
    """Group contiguous tokens whose hierarchical label == target_code."""
    spans = []
    current = []

    for tok, lab in zip(tokens, labels):
        if lab == target_code:
            current.append(tok)
        else:
            if current:
                spans.append(" ".join(current))
                current = []

    if current:
        spans.append(" ".join(current))

    return spans


def get_doc_ids(split):
    # Use the participants starting_spans folder as the id list — all
    # three elements are annotated for the same doc set per split.
    folder = os.path.join(
        DATA_DIR, "annotations", "aggregated", "starting_spans",
        "participants", split
    )
    return sorted(
        f.split(".")[0] for f in os.listdir(folder) if f.endswith(".ann")
    )


def load_abstract_text(pmid):
    path = os.path.join(DATA_DIR, "documents", f"{pmid}.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def main():
    ensure_extracted()

    doc_ids = get_doc_ids(SPLIT)
    print(f"Found {len(doc_ids)} documents in split '{SPLIT}'")

    written = 0
    skipped = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        for pmid in doc_ids:

            try:
                tokens = load_tokens(pmid)
                abstract_text = load_abstract_text(pmid)
            except FileNotFoundError:
                skipped += 1
                continue

            raw_entities = {}
            ok = True

            for element, letter in ELEMENT_TO_LETTER.items():
                labels = load_labels(element, SPLIT, pmid, phase="starting_spans")
                if labels is None or len(labels) != len(tokens):
                    ok = False
                    break
                raw_entities[letter] = spans_from_binary_labels(tokens, labels)

            if not ok:
                skipped += 1
                continue

            # Bonus: hierarchical "Control" (7) sub-spans within Intervention
            hier_labels = load_labels(
                "interventions", SPLIT, pmid, phase="hierarchical_labels"
            )
            comparator_hierarchical = []
            if hier_labels is not None and len(hier_labels) == len(tokens):
                comparator_hierarchical = spans_from_code_labels(
                    tokens, hier_labels, target_code="7"
                )

            record = {
                "id": pmid,
                "abstract": abstract_text,
                "raw_entities": raw_entities,
                "comparator_hierarchical": comparator_hierarchical,
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Converted: {written}")
    print(f"Skipped (missing/misaligned files): {skipped}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

