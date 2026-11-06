"""Parses the 'Population: ...\\nIntervention: ...' style model output and scores
it against gold spans with set-based precision / recall / F1 per PICO element.
Same idea as AlpaPICO's metric.py (tp/fp/fn over the span sets per element),
implemented independently here so this project has no dependency on that repo.
"""
import re
import string

ELEMENT_ORDER = ["Population", "Intervention", "Comparator", "Outcome"]

_LABEL_PATTERN = re.compile(
    r"(Population|Intervention|Comparator|Outcome):\s*(.*?)(?=\n\s*(?:Population|Intervention|Comparator|Outcome):|\Z)",
    re.DOTALL,
)

_DASH_PATTERN = re.compile(r"\s*-\s*")


def _normalize_span(span):
    """Canonicalize a span for exact-match comparison only -- callers that need
    the original text for display/logging should keep using the raw span from
    parse_model_output(); this is applied at scoring time so it affects both
    Phase 3's training reward and Phase 5's eval F1 identically.

    - lowercase
    - collapse any spaced-out hyphen ("preterm - infants", "preterm- infants",
      "preterm -infants") to a bare hyphen ("preterm-infants")
    - strip trailing punctuation/whitespace (trailing periods, commas, etc.)

    This absorbs formatting differences that aren't real extraction errors, so
    reward/F1 changes can be attributed to the policy rather than to
    inconsistent surface punctuation between predicted and gold spans.
    """
    s = span.lower().strip()
    s = _DASH_PATTERN.sub("-", s)
    s = s.rstrip(string.punctuation + " ")
    return s


def _normalize_spans(spans):
    return {_normalize_span(s) for s in spans if s}


def parse_model_output(text):
    """Parse a model completion into {element: [span, ...]}. Unparseable or
    'none' entries come back as an empty list for that element. Spans within
    a line are split on '|' (not ',') since a span's own text often contains
    commas (e.g. "healthy, full-term infants")."""
    spans = {name: [] for name in ELEMENT_ORDER}
    for label, value in _LABEL_PATTERN.findall(text or ""):
        value = value.strip()
        if not value or value.lower().startswith("none"):
            continue
        parts = [p.strip() for p in value.split("|")]
        spans[label] = [p for p in parts if p]
    return spans


def _counts_one(pred, gold):
    counts = {}
    for name in ELEMENT_ORDER:
        pred_set, gold_set = _normalize_spans(pred.get(name, [])), _normalize_spans(gold.get(name, []))
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        counts[name] = (tp, fp, fn)
    return counts


def aggregate_metrics(pred_list, gold_list):
    """Corpus-level precision/recall/F1 per element, plus a macro-averaged 'overall' row."""
    totals = {name: [0, 0, 0] for name in ELEMENT_ORDER}
    for pred, gold in zip(pred_list, gold_list):
        for name, (tp, fp, fn) in _counts_one(pred, gold).items():
            totals[name][0] += tp
            totals[name][1] += fp
            totals[name][2] += fn

    results, f1s = {}, []
    for name in ELEMENT_ORDER:
        tp, fp, fn = totals[name]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        results[name] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        f1s.append(f1)
    results["overall"] = {"f1": sum(f1s) / len(f1s)}
    return results


def reward_f1_element(pred, gold, element):
    """Scalar reward for a single PICO element (e.g. just 'Population'), used
    while the policy is only being trained/rewarded on one element at a time."""
    pred_set, gold_set = _normalize_spans(pred.get(element, [])), _normalize_spans(gold.get(element, []))
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    if tp == fp == fn == 0:
        return 1.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def reward_f1_element_shaped(pred, gold, element, shaping_weight=0.1):
    """TRAINING-ONLY reward: reward_f1_element blended with a small dense
    "did the model correctly decide whether this element has any span at
    all" bonus. Exact-match span F1 alone is nearly binary for Population --
    most docs have exactly one long gold span, so any wording difference in
    a sampled completion scores tp=0 and the reward collapses to 0 for that
    sample; a whole sampled group can land on all-0 (or all-1) rewards, i.e.
    zero variance -- no gradient signal for that document (see the
    `n_degenerate` counter in phase3_grpo_train.py). The has-a-span bonus
    doesn't require an exact string match, so it can still differ across a
    group of slightly different completions even when none of them match
    verbatim, giving GRPO something to learn from on those documents.

    Stays in [0, 1] (a weighted average of two [0, 1] terms). NOT used for
    evaluation -- Phase 5's aggregate_metrics and the plain reward_f1_element
    above are untouched, so this never affects a reported F1 number, only
    the RL reward signal used to update the policy.
    """
    base = reward_f1_element(pred, gold, element)
    pred_has = bool(_normalize_spans(pred.get(element, [])))
    gold_has = bool(_normalize_spans(gold.get(element, [])))
    has_match = 1.0 if pred_has == gold_has else 0.0
    return (1 - shaping_weight) * base + shaping_weight * has_match


def reward_f1(pred, gold):
    """Scalar reward used by GRPO: macro-F1 across the four elements for one document.
    A document where both pred and gold are empty for an element counts as correct
    for that element (reward 1.0), so the policy isn't punished for correctly
    predicting 'nothing here'."""
    counts = _counts_one(pred, gold)
    f1s = []
    for name in ELEMENT_ORDER:
        tp, fp, fn = counts[name]
        if tp == fp == fn == 0:
            f1s.append(1.0)
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return sum(f1s) / len(f1s)
