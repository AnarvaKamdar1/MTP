"""Prompt construction with exact steered-span localisation.

The span is located by character offsets rather than by re-tokenising the
instruction text and searching for a matching id sequence. Standalone
re-tokenisation drifts from in-context tokenisation (leading-space merges,
newlines, punctuation), and on a multi-hundred-token rules block it can fail
silently and return an empty span.
"""

from typing import List, Tuple

from guidelines import ELEMENTS, GENERAL_RULES


# Sentinels: unique strings stripped out before the prompt is tokenised
_OPEN = "<<<STEER>>>"
_CLOSE = "<<</STEER>>>"


def _build_user_message(abstract: str, element: str, cfg) -> str:
    """Task first, output format second, steered instruction block last."""
    meta = ELEMENTS[element]

    task = (
        f"You are a biomedical annotator extracting the {meta['name']} "
        f"({meta['letter']}) elements from the clinical trial abstract below.\n\n"
        f"Abstract:\n{abstract.strip()}\n"
    )

    fmt = (
        f'\nReturn a single JSON object with exactly one key "{element}", whose value is a '
        f"list of strings. Each string must be a span copied verbatim from the abstract. "
        f"Use an empty list if no such element is present. Output the JSON object only, "
        f"with no explanation before or after it.\n"
    )

    rules = f"{meta['definition']}\n\n{meta['rules']}"
    if cfg.steer_general_rules:
        rules = f"{GENERAL_RULES}\n\n{rules}"

    instruction = (
        "\nYour response must follow the annotation guidelines below:\n"
        f"{_OPEN}{rules}{_CLOSE}\n"
    )

    if cfg.steer_output_format:
        return task + f"{_OPEN}{fmt}{_CLOSE}" + instruction.replace(_OPEN, "").replace(_CLOSE, "")
    return task + fmt + instruction


def build_element_prompt(abstract: str, element: str, tokenizer, cfg):
    """Builds chat prompt and returns (input_ids, [(start_token, end_token_exclusive)])."""
    user_msg = _build_user_message(abstract, element, cfg)

    # Render chat template to string so we can measure character offsets
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Strip sentinels while recording exact character start and end positions
    clean_parts, char_spans, cursor = [], [], 0
    text, out_len = templated, 0
    while True:
        i = text.find(_OPEN, cursor)
        if i == -1:
            clean_parts.append(text[cursor:])
            break
        j = text.find(_CLOSE, i)
        if j == -1:
            raise ValueError("Unbalanced steering sentinel in prompt.")
        head = text[cursor:i]
        body = text[i + len(_OPEN):j]
        clean_parts.append(head)
        out_len += len(head)
        char_spans.append((out_len, out_len + len(body)))
        clean_parts.append(body)
        out_len += len(body)
        cursor = j + len(_CLOSE)

    clean = "".join(clean_parts)

    # Fast tokenizer returns character-to-token offset mappings
    enc = tokenizer(
        clean,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = enc["offset_mapping"][0].tolist()

    # Map character spans to token indices
    token_spans: List[Tuple[int, int]] = []
    for cs, ce in char_spans:
        idx = [
            t for t, (s, e) in enumerate(offsets)
            if s < ce and e > cs and e > s  # Overlapping non-empty tokens
        ]
        if not idx:
            raise ValueError(f"Could not localise steered span for '{element}'.")
        token_spans.append((idx[0], idx[-1] + 1))  # End-exclusive token index

    total = enc["input_ids"].shape[1]
    # Prefill runs on input_ids[:, :-1], so the steered span must reside inside prefill
    if max(e for _, e in token_spans) > total - 1:
        raise ValueError("Steered span runs into the last prompt token.")

    if cfg.verbose:
        covered = sum(e - s for s, e in token_spans)
        print(f"[{element}] prompt={total} tok, steered={covered} tok, spans={token_spans}")

    return enc["input_ids"], token_spans