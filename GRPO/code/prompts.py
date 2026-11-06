"""Prompt construction for both models, built directly from guidelines.py so the
rules stay in one place. Both prompts ask for the same structured output format:

    Population: span one | span two
    Intervention: span three
    Comparator: none
    Outcome: span four | span five

Spans are separated by " | " (not a comma) because clinical spans routinely
contain their own internal commas (e.g. "healthy, full-term infants") that
would otherwise get mis-split into two spans. metrics.parse_model_output()
reads this format back.
"""
from guidelines import GENERAL_RULES, ELEMENTS, ELEMENT_ORDER


def _guideline_block():
    lines = [GENERAL_RULES, ""]
    for name in ELEMENT_ORDER:
        info = ELEMENTS[name]
        lines.append(f"{info['letter']} - {info['name']}: {info['definition']}")
        lines.append(info["rules"])
        lines.append("")
    return "\n".join(lines)


GUIDELINE_BLOCK = _guideline_block()

_OUTPUT_FORMAT = (
    "Respond in exactly this format, one line per element, spans copied verbatim\n"
    "from the abstract and separated by ' | ' (a pipe, not a comma -- spans can contain "
    "their own commas) -- write 'none' if an element has no spans:\n"
    "Population: ...\nIntervention: ...\nComparator: ...\nOutcome: ..."
)

# The small model's policy is currently only trained on ONE element -- change this
# to switch which element Phase 1 proposes keywords for and Phase 3 rewards on.
KEYWORD_ELEMENT = "Population"


def keyword_prompt(text, element=KEYWORD_ELEMENT):
    """Phase 1 / Phase 4 prompt: the small model proposes candidate keyword spans
    for a single PICO element (only `element`'s definition/rules are included, so
    the policy isn't asked to reason about elements it isn't being trained on)."""
    info = ELEMENTS[element]
    return (
        "You are assisting a PICO information-extraction system for biomedical abstracts.\n"
        f"Read the guideline below, then the abstract, and propose short candidate KEYWORD "
        f"spans (copied verbatim from the abstract) for the {info['name']} ({element}) "
        f"element only. These are hints for a downstream extractor -- do not explain your "
        f"answer, only list spans.\n\n"
        f"{GENERAL_RULES}\n\n"
        f"{info['letter']} - {info['name']}: {info['definition']}\n{info['rules']}\n\n"
        f"Abstract:\n{text}\n\n"
        "Respond in exactly this format (spans copied verbatim from the abstract, separated "
        f"by ' | ' -- a pipe, not a comma, since a span may contain its own commas -- write "
        f"'none' if there are no spans):\n{element}: ..."
    )


def extraction_prompt(text, keyword_hints=None, hint_element=KEYWORD_ELEMENT):
    """Phase 2 / Phase 4 prompt: the large model performs the full P/I/C/O extraction,
    optionally guided by single-element keyword hints from the small model."""
    hints_block = ""
    if keyword_hints:
        hints_block = (
            f"\nA smaller assistant model proposed these candidate {hint_element} spans as "
            "a starting point, based on the same guidelines:\n" + keyword_hints.strip() + "\n"
            f"\nWeigh them alongside your own reading of the abstract -- neither defer to them "
            f"automatically nor set them aside by default. Keep any that hold up against the "
            f"guidelines and the abstract, correct or drop any that don't, and add any genuine "
            f"{hint_element} spans they missed.\n"
        )
    return (
        "You are a careful biomedical annotator performing PICO information extraction.\n\n"
        f"{GUIDELINE_BLOCK}\n"
        f"Abstract:\n{text}\n"
        f"{hints_block}\n"
        f"Extract every span for each element, copied verbatim from the abstract, following "
        f"the rules above exactly. {_OUTPUT_FORMAT}"
    )
