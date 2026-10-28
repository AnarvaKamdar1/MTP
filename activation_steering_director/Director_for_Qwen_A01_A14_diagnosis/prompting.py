"""Prompt construction with exact steered-span and activation localisation."""

from typing import Dict, List, Tuple

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


def _char_to_token_span(offsets, char_start: int, char_end: int):
    idx = [
        t for t, (s, e) in enumerate(offsets)
        if s < char_end and e > char_start and e > s
    ]
    if not idx:
        return None
    return idx[0], idx[-1] + 1


def build_element_prompt(abstract: str, element: str, tokenizer, cfg):
    """Build prompt and return token spans needed by DIRECTER + activation analysis.

    Returns:
        input_ids: [1, seq]
        token_spans: steered KV spans
        activation_spans: A01-A12 token spans in the prompt. A13/A14 are filled by
            the generator because they depend on the accepted steered probe.
    """
    abstract_clean = abstract.strip()
    user_msg = _build_user_message(abstract, element, cfg)

    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=cfg.enable_thinking,
    )

    # Strip sentinels while recording exact character start/end positions.
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

    enc = tokenizer(
        clean,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = enc["offset_mapping"][0].tolist()
    total = enc["input_ids"].shape[1]

    token_spans: List[Tuple[int, int]] = []
    for cs, ce in char_spans:
        span = _char_to_token_span(offsets, cs, ce)
        if span is None:
            raise ValueError(f"Could not localise steered span for '{element}'.")
        token_spans.append(span)

    if max(e for _, e in token_spans) > total - 1:
        raise ValueError("Steered span runs into the last prompt token.")

    # Locate the abstract in the final chat-template string. The prompt itself is
    # unchanged; this is only metadata for activation extraction.
    abstract_marker = "Abstract:\n"
    abstract_char_start = clean.find(abstract_marker)
    if abstract_char_start == -1:
        raise ValueError("Could not find the abstract marker in the rendered prompt.")
    abstract_char_start += len(abstract_marker)
    abstract_char_end = abstract_char_start + len(abstract_clean)
    abstract_span = _char_to_token_span(
        offsets, abstract_char_start, abstract_char_end
    )
    if abstract_span is None:
        raise ValueError("Could not localise abstract token span.")

    # Each task-specific prompt contains only its own meta['name'] and meta['letter'].
    # Therefore only the corresponding A01/A02, A03/A04, A05/A06, or A07/A08 pair
    # exists in a given prompt. Do not search for the other three task labels.
    task_activation_ids = {
        "Population": ("A01", "A02"),
        "Intervention": ("A03", "A04"),
        "Comparator": ("A05", "A06"),
        "Outcome": ("A07", "A08"),
    }
    meta = ELEMENTS[element]
    name_aid, letter_aid = task_activation_ids[element]
    requested_labels = [
        (name_aid, meta["name"]),
        (letter_aid, f"({meta['letter']})"),
    ]

    activation_spans: Dict[str, Tuple[int, int]] = {}
    activation_text: Dict[str, str] = {}
    for aid, phrase in requested_labels:
        # Search from the task-label region, before the abstract, so labels are not
        # accidentally picked from the abstract itself.
        char_idx = clean.find(phrase, 0, abstract_char_start)
        if char_idx == -1:
            raise ValueError(f"Could not locate activation label '{phrase}' for {aid}.")
        span = _char_to_token_span(offsets, char_idx, char_idx + len(phrase))
        if span is None:
            raise ValueError(f"Could not tokenise activation label '{phrase}' for {aid}.")
        activation_spans[aid] = span
        activation_text[aid] = phrase

    # A09 is the final prompt token that is fed separately after prefill. In the
    # Qwen3 chat template this is the generation-start context token immediately
    # before the model predicts the first generated token.
    activation_spans["A09"] = (total - 1, total)
    activation_text["A09"] = tokenizer.decode(enc["input_ids"][0, total - 1:total])

    # A10 = last token belonging to the user abstract, before the format/instruction
    # text and chat-template assistant generation marker.
    activation_spans["A10"] = (abstract_span[1] - 1, abstract_span[1])
    activation_text["A10"] = tokenizer.decode(
        enc["input_ids"][0, abstract_span[1] - 1:abstract_span[1]]
    )

    # A11 = entire rendered input; A12 = abstract-only local representation.
    activation_spans["A11"] = (0, total)
    activation_spans["A12"] = abstract_span
    activation_text["A11"] = "<full_input_sequence>"
    activation_text["A12"] = "<abstract_span>"

    if cfg.verbose:
        covered = sum(e - s for s, e in token_spans)
        print(
            f"[{element}] prompt={total} tok, steered={covered} tok, "
            f"steered_spans={token_spans}, abstract_span={abstract_span}"
        )

    return enc["input_ids"], token_spans, {
        "spans": activation_spans,
        "text": activation_text,
        "token_ids": enc["input_ids"][0].tolist(),
    }
