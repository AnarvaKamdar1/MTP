"""Parsing utilities for the PICO CoNLL-style BIO dataset (train.txt / test.txt).

Expected format (tab-separated, one token per line, blank line = sentence break):

    -DOCSTART-\tO
    <blank>
    Title\tO
    :\tO
    <title tokens...>\t<BIO tags...>
    <blank>
    METHODS\tO
    :\tO
    <methods tokens...>\t<BIO tags...>
    <blank>            (may repeat for several methods sentences)
    -DOCSTART-\tO
    ...

Tags are "O" (outside any span) or "B-<L>" / "I-<L>" where <L> in {P, I, C, O}
matches the element letters used in guidelines.py (Population, Intervention,
Comparator, Outcome). Note "O" alone means "no entity", which is distinct
from the letter "O" used for the Outcome element in "B-O"/"I-O" tags.
"""
from dataclasses import dataclass, field

LETTER_TO_ELEMENT = {"P": "Population", "I": "Intervention", "C": "Comparator", "O": "Outcome"}
ELEMENT_ORDER = ["Population", "Intervention", "Comparator", "Outcome"]

_NO_SPACE_BEFORE = set(list(",.;:!?)]}%"))
_NO_SPACE_AFTER = set(list("([{"))


def detokenize(tokens):
    """Cheap heuristic detokenizer, good enough for building LLM prompts."""
    out = []
    for i, tok in enumerate(tokens):
        if i > 0:
            prev = tokens[i - 1]
            if tok in _NO_SPACE_BEFORE or prev in _NO_SPACE_AFTER or tok.startswith("'"):
                pass
            else:
                out.append(" ")
        out.append(tok)
    return "".join(out)


def bio_to_spans(tokens, tags):
    """Decode a BIO tag sequence into {element_name: [span_text, ...]}."""
    spans = {name: [] for name in ELEMENT_ORDER}
    current_letter, current_tokens = None, []

    def flush():
        if current_letter and current_tokens:
            element = LETTER_TO_ELEMENT.get(current_letter)
            if element:
                spans[element].append(detokenize(current_tokens))

    for tok, tag in zip(tokens, tags):
        if tag == "O" or tag == "":
            flush()
            current_letter, current_tokens = None, []
            continue
        prefix, _, letter = tag.partition("-")
        if prefix == "B" or letter != current_letter:
            flush()
            current_letter, current_tokens = letter, [tok]
        else:  # "I-" continuing the same entity
            current_tokens.append(tok)
    flush()
    return spans


@dataclass
class PicoDoc:
    doc_id: int
    title_tokens: list = field(default_factory=list)
    title_tags: list = field(default_factory=list)
    methods_sentences: list = field(default_factory=list)  # list of (tokens, tags)

    @property
    def text(self):
        title = detokenize(self.title_tokens)
        methods = " ".join(detokenize(t) for t, _ in self.methods_sentences)
        return f"Title: {title}\n\nMETHODS: {methods}"

    @property
    def gold_spans(self):
        all_tokens, all_tags = list(self.title_tokens), list(self.title_tags)
        for toks, tags in self.methods_sentences:
            all_tokens += toks
            all_tags += tags
        return bio_to_spans(all_tokens, all_tags)


def load_conll(path):
    """Parse a train.txt / test.txt file into a list of PicoDoc."""
    with open(path, encoding="utf-8") as f:
        raw_lines = f.read().split("\n")

    docs = []
    doc_id = -1
    section = None  # "title" | "methods" | None
    title_tokens, title_tags = [], []
    methods_sentences = []
    cur_tokens, cur_tags = [], []

    def flush_sentence():
        nonlocal cur_tokens, cur_tags
        if cur_tokens:
            methods_sentences.append((cur_tokens, cur_tags))
        cur_tokens, cur_tags = [], []

    def flush_doc():
        nonlocal title_tokens, title_tags, methods_sentences
        flush_sentence()
        if title_tokens or methods_sentences:
            docs.append(PicoDoc(doc_id, title_tokens, title_tags, methods_sentences))
        title_tokens, title_tags, methods_sentences = [], [], []

    for line in raw_lines:
        line = line.rstrip("\n")
        if not line.strip():
            if section == "methods":
                flush_sentence()
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        token, tag = parts[0], parts[1]

        if token == "-DOCSTART-":
            flush_doc()
            doc_id += 1
            section = None
            continue
        if token == "Title" and tag == "O" and section is None:
            section = "title"
            continue
        if token == "METHODS" and tag == "O":
            flush_sentence()
            section = "methods"
            continue
        if token == ":" and tag == "O" and (
            (section == "title" and not title_tokens)
            or (section == "methods" and not cur_tokens and not methods_sentences)
        ):
            continue  # skip the header colon after "Title" / "METHODS"

        if section == "title":
            title_tokens.append(token)
            title_tags.append(tag)
        elif section == "methods":
            cur_tokens.append(token)
            cur_tags.append(tag)
    flush_doc()
    return docs


if __name__ == "__main__":
    import sys
    docs = load_conll(sys.argv[1] if len(sys.argv) > 1 else "data/train.txt")
    print(f"Parsed {len(docs)} documents")
    print(docs[0].text)
    print(docs[0].gold_spans)
