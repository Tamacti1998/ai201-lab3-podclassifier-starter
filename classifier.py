import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def _shorten(text: str, max_chars: int = 200) -> str:
    """
    Trim an example description to ~max_chars, cutting at a sentence boundary
    when possible so the excerpt stays readable. Keeps token cost down without
    losing the structural cue (which is almost always stated early).
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    # Prefer ending at the last sentence break inside the window.
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if cut >= max_chars // 2:
        return window[:cut + 1]
    # Otherwise fall back to the last word boundary.
    space = window.rfind(" ")
    return (window[:space] if space > 0 else window).rstrip() + "…"


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.

    TODO — Milestone 2:

    Your prompt needs to:
      1. Describe the task and the four valid labels
      2. Show the labeled training examples so the LLM can learn the pattern
      3. Present the new description and ask for a classification

    The LLM should return a single label from VALID_LABELS (exactly as written)
    plus a brief explanation of its reasoning. Think carefully about the output
    format you request — you'll need to parse it in classify_episode().

    Before writing code, complete specs/classifier-spec.md.
    """
    instructions = (
        "You are classifying podcast episodes by their format. Classify the "
        "episode into exactly one of these four labels:\n\n"
        "- interview: a conversation between a host and one or more guests\n"
        "- solo: a single host speaking from memory, experience, or opinion — "
        "no guests, no assembled external sources\n"
        "- panel: multiple guests with roughly equal speaking time, often "
        "debating or discussing a topic together\n"
        "- narrative: a story assembled from external sources — interviews, "
        "archival audio, reporting — with a clear narrative arc\n\n"
        "Classify by the structural format, not the topic or tone. "
        "Return only the label and your reasoning. Do not explain the taxonomy."
    )

    # Output format request — parsed by anchor in classify_episode().
    output_format = (
        "Respond in exactly this format, with Label on its own line first:\n\n"
        "Label: <one of: interview, solo, panel, narrative>\n"
        "Reasoning: <one or two sentences>\n\n"
        "Use one of the four label strings verbatim, with nothing else on the "
        "Label line."
    )

    parts = [instructions]

    # Edge case: no labeled examples -> fall back to a zero-shot prompt.
    if labeled_examples:
        parts.append("Here are labeled examples:")
        example_blocks = []
        for ex in labeled_examples:
            # Truncate example descriptions to keep the prompt cheap — the
            # format signal lives in the first sentence or two, and the full
            # blurbs blow through the daily token budget across 20 calls.
            example_blocks.append(
                f"Title: {ex.get('title', '')}\n"
                f"Description: {_shorten(ex.get('description', ''))}\n"
                f"Label: {ex['label']}"
            )
        parts.append("\n\n---\n\n".join(example_blocks))

    # The new episode to classify — same format, Label withheld.
    parts.append(
        "Now classify this episode:\n\n"
        f"Description: {description}\n"
        "Label: ?"
    )

    # Always end with the explicit output-format instruction.
    parts.append(output_format)

    return "\n\n".join(parts)


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.

    TODO — Milestone 2 (complete after build_few_shot_prompt):

    Steps:
      1. Call build_few_shot_prompt() to construct the prompt
      2. Send it to the LLM via _client.chat.completions.create()
      3. Parse the response to extract a label and reasoning
      4. Validate the label — if it's not in VALID_LABELS, set it to "unknown"
      5. Return a dict with "label" and "reasoning" keys

    Handle the case where the LLM returns something unparseable gracefully —
    don't let a bad response crash the whole evaluation.

    Before writing code, complete specs/classifier-spec.md.
    """
    # Step 1 — build the prompt from the labeled examples + this description.
    prompt = build_few_shot_prompt(labeled_examples, description)

    # Step 2 — send to the LLM. Step 5 — wrap so one bad call can't crash the
    # 20-call evaluation loop; any failure returns label "unknown".
    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
        )
        text = response.choices[0].message.content or ""
    except Exception as e:  # network / API / auth / rate-limit errors
        return {"label": "unknown", "reasoning": f"API error: {e}"}

    # Step 3 — anchor parse on the "Label:" / "Reasoning:" lines.
    label, reasoning = _parse_response(text)

    # Step 4 — validate against VALID_LABELS; anything else becomes "unknown"
    # but we keep the reasoning so off-format responses can be inspected.
    if label not in VALID_LABELS:
        reasoning = reasoning or text.strip() or "No reasoning returned."
        label = "unknown"

    return {"label": label, "reasoning": reasoning}


def _strip_markdown(s: str) -> str:
    """Strip leading/trailing markdown noise (*, `, #) and quotes/space."""
    return s.strip().strip("*`#").strip().strip('"').strip("'").strip()


def _parse_response(text: str) -> tuple[str, str]:
    """
    Extract (label, reasoning) from the LLM response using anchor parsing.

    Primary: find the line starting with "label:" and the line starting with
    "reasoning:" (case-insensitive, markdown stripped). Reasoning may span the
    remaining lines. Fallback: scan the whole text for any valid label keyword.
    Returns ("", "") for label/reasoning if nothing usable is found.
    """
    lines = text.splitlines()
    label = ""
    reasoning = ""

    for i, line in enumerate(lines):
        cleaned = _strip_markdown(line).lower()
        if not label and cleaned.startswith("label:"):
            label = _strip_markdown(line.split(":", 1)[1]).lower()
        elif cleaned.startswith("reasoning:"):
            # Take the rest of this line plus any continuation lines.
            first = line.split(":", 1)[1].strip()
            rest = [l.strip() for l in lines[i + 1:] if l.strip()]
            reasoning = " ".join([first, *rest]).strip()

    # Fallback: no "Label:" line — scan the whole response for a known label.
    if not label:
        lowered = text.lower()
        for valid in VALID_LABELS:
            if valid in lowered:
                label = valid
                break

    if not reasoning:
        reasoning = text.strip()

    return label, reasoning
