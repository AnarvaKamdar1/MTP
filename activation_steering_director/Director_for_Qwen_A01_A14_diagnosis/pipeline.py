"""PICO extraction pipeline: 4 steered runs per abstract merged into one JSON."""

import json
from typing import Dict, List, Optional

from directer import DirecterGenerator
from guidelines import ELEMENT_ORDER, ELEMENTS
from prompting import build_element_prompt


def _extract_json_object(text: str) -> dict:
    """Extracts the first balanced JSON object {...} from raw model output."""
    start = text.find("{")
    if start == -1:
        return {}
    depth, in_str, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def _coerce_spans(value) -> List[str]:
    """Normalizes extracted model values into a clean list of unique string spans."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []

    out, seen = [], set()
    for item in value:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


class PICODirecterPipeline:
    def __init__(self, model, tokenizer, cfg):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.generator = DirecterGenerator(model, tokenizer, cfg)

    def extract(self, abstract: str, return_raw: bool = False, doc_id: Optional[str] = None) -> Dict:
        result, raw = {}, {}

        for element in ELEMENT_ORDER:
            input_ids, spans, activation_meta = build_element_prompt(
                abstract, element, self.tokenizer, self.cfg
            )
            # element_letter (p/i/c/o) + doc_id identify the activation bundle.
            element_letter = ELEMENTS[element]["letter"].lower()
            text = self.generator.generate(
                input_ids,
                spans,
                activation_meta=activation_meta,
                doc_tag=doc_id,
                element_letter=element_letter,
            )
            raw[element] = text

            obj = _extract_json_object(text)
            # Accept exact key or single-key fallback
            value = obj.get(element)
            if value is None and len(obj) == 1:
                value = next(iter(obj.values()))

            result[element] = _coerce_spans(value)

            if self.cfg.verbose:
                print(f"[{element}] -> Found {len(result[element])} span(s)")

        return {"pico": result, "raw": raw} if return_raw else result