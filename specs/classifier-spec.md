# Classifier Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 2.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `build_few_shot_prompt()` and
`classify_episode()` in `classifier.py`.

---

## build_few_shot_prompt(labeled_examples, description)

### What it does
Constructs a prompt string for the LLM that includes the task instructions,
all labeled training examples, and the new episode description to classify.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `labeled_examples` | `list[dict]` | Each dict has `"title"`, `"description"`, `"label"` (and others). These are the examples you labeled in Milestone 1. |
| `description` | `str` | The episode description to classify. |

### Output

| Return value | Type | Description |
|---|---|---|
| prompt | `str` | A complete prompt string ready to send to the LLM. |

---

### Spec fields — fill these in before writing code

**Task instruction (what should the LLM know about the task?):**

```
You are classifying podcast episodes by their format. Classify the episode
into exactly one of these four labels:

- interview: a conversation between a host and one or more guests
- solo: a single host speaking from memory, experience, or opinion — no guests,
  no assembled external sources
- panel: multiple guests with roughly equal speaking time, often debating or
  discussing a topic together
- narrative: a story assembled from external sources — interviews, archival
  audio, reporting — with a clear narrative arc

Return only the label and your reasoning. Do not explain the taxonomy.
```

---

**How should labeled examples be formatted in the prompt?**

```
Each example should include the episode title, a brief excerpt or the full
description, and the correct label. Separate examples with a blank line or
a delimiter like "---". Include all fields that help the model see why the
label was applied — title and description are both useful; other fields
(like episode ID) are not needed.
```

---

**Example block sketch (write one concrete example):**

```
Title: {title}
Description: {description}
Label: {label}
```

---

**How should the new episode (to be classified) be presented?**

```
Present it in the same format as the labeled examples, but omit the Label
line and replace it with an instruction to classify. For example:

Title: {title}
Description: {description}
Label: ?

Then add a line like: "Classify the episode above. Return your answer in
the format below:" followed by the output format you chose.
```

---

**What output format should you request from the LLM?**

```
CHOICE: Two prefixed lines —

    Label: <one of: interview | solo | panel | narrative>
    Reasoning: <one or two sentences>

Request exactly this shape and instruct the model to put Label first, on its
own line, and to use one of the four strings verbatim with nothing else on
that line.

Why this over the alternatives:

- "Label: X / Reasoning: Y" (CHOSEN): parsed by ANCHOR, not position — scan
  for the line starting with "label:" (case-insensitive, after stripping
  markdown like ** or #). Resilient to preamble/trailing prose, and degrades
  gracefully: if the Reasoning line is missing or malformed, the label is
  still recoverable on its own. Multi-line reasoning is fine because it is the
  last field. Main weakness — markdown bolding / capitalization drift — is
  handled by normalizing (strip, lower, strip "*#").

- JSON object: unambiguous WHEN valid, but this model (llama-3.3-70b, no
  enforced JSON mode) often wraps it in ```json fences, adds a preamble line,
  or emits invalid JSON (trailing commas, unescaped quotes). It is all-or-
  nothing: one malformed brace loses BOTH fields. Would need substring
  extraction plus try/except just to match the robustness we get for free
  above. Rejected.

- Bare label on line 1, then prose: simplest parse, but most fragile — any
  preamble ("Sure, here's my classification:") becomes the "label," and there
  is no anchor to recover from drift. Rejected.
```

---

**Edge cases to handle in the prompt:**

```
- labeled_examples is empty: still produce a valid prompt. Fall back to a
  zero-shot prompt — the task instruction plus the four label definitions are
  enough for the model to classify without examples. Do NOT crash or return ""
  (an empty prompt would waste an API call and guarantee an "unknown").
- Very short / thin description: don't special-case it. Still ask for the same
  output format; the model should pick the best-fit label from the definitions.
  Low-signal inputs are exactly when the Reasoning field is most useful for
  debugging, so always request it.
- Whatever the input, the prompt must always end with the explicit output-format
  instruction so the response stays parseable.
```

---

## classify_episode(description, labeled_examples)

### What it does
Classifies a single podcast episode description using the few-shot LLM classifier.
Returns a dict with a label and reasoning.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `description` | `str` | The episode description to classify. |
| `labeled_examples` | `list[dict]` | Labeled training examples from `load_labeled_examples()`. |

### Output

| Return value | Type | Description |
|---|---|---|
| result | `dict` | Must have keys `"label"` and `"reasoning"`. `"label"` must be one of `VALID_LABELS` or `"unknown"`. |

---

### Spec fields — fill these in before writing code

**Step 1 — Build the prompt:**

```
Call build_few_shot_prompt(labeled_examples, description) and store the
returned string in a variable (e.g., prompt). Pass through both arguments
exactly as received — no modification needed before calling.
```

---

**Step 2 — Send to the LLM:**

```
Call _client.chat.completions.create() with:
  - model: the model name from config (LLM_MODEL)
  - messages: a list with one dict — {"role": "user", "content": prompt}
    (system-design.md shows an optional system message too — either shape works)
  - max_tokens: a reasonable limit (e.g., 200–300) to keep responses concise

Extract the response text from:
  response.choices[0].message.content
```

---

**Step 3 — Parse the response:**

```
Matches the chosen "Label: X / Reasoning: Y" format. Parse by anchor:

  1. Split the response text into lines.
  2. Find the line whose stripped, lowercased, markdown-stripped form starts
     with "label:". Take the text after the colon as the raw label; normalize
     it (strip, lower, remove surrounding "*`#" and quotes).
  3. Find the line starting with "reasoning:"; take the text after the colon.
     If reasoning spans multiple lines, join the remainder of the response
     after the Reasoning: line. If no Reasoning line is found, fall back to the
     full response text (minus the Label line) so reasoning is never empty.
  4. If NO line starts with "label:", fall back: scan the whole response for
     the first occurrence of any VALID_LABELS string. If still nothing, the
     label is unparseable -> handled in Step 4 as "unknown".
```

---

**Step 4 — Validate the label:**

```
After normalizing the parsed label (strip + lower), check membership in
VALID_LABELS. If it is one of the four, keep it. If it is anything else —
a synonym, a sentence, empty, or unparseable — set label to "unknown".
Keep the parsed reasoning even when the label is "unknown", since it shows
why the model went off-format and aids debugging. "unknown" is a valid
output per the contract; it is counted as a miss in evaluation, not an error.
```

---

**Step 5 — Handle errors gracefully:**

```
Wrap the API call and parsing in try/except so one failure can't crash the
20-call evaluation loop. Failure modes and responses:

- Network / API error (timeout, rate limit, auth, 5xx): catch the exception,
  return {"label": "unknown", "reasoning": "API error: <short message>"}.
- Empty or None response content: treat as unparseable -> label "unknown",
  reasoning notes the empty response.
- Unparseable text (no label found, see Step 3 fallback): label "unknown",
  reasoning = the raw response (truncated) so it can be inspected.

The function ALWAYS returns a dict with "label" and "reasoning" keys and never
raises. "unknown" is the single sentinel for every non-success path, so the
caller never has to distinguish error types to keep going.
```

---

### Return value structure

```python
{
    "label": str,      # one of VALID_LABELS, or "unknown" if invalid/error
    "reasoning": str,  # brief explanation from the LLM
}
```

---

## Notes on label quality

The classifier is only as good as your labels. If your training examples have
inconsistent or ambiguous labels, the LLM will learn the wrong pattern.

Before implementing the classifier, re-read `data/taxonomy.md` and double-check
any labels you're unsure about. Annotation quality is part of the lab.

---

## Implementation Notes

*Captured from a real Groq call (model: llama-3.3-70b-versatile) using the
implemented build_few_shot_prompt() with all 20 labeled examples. NOTE:
classify_episode() itself is still a stub at the time of writing, so the parse
described below is the specced approach run by hand against this real response —
not yet wired into the function.*

**Test: what does the raw LLM response look like for one episode?**

```
Episode tested: "Dr. Priya Nair on Adolescent Mental Health After the Pandemic"
  (the interview example from app.py)

Raw response text (verbatim, between the markers):
---
Label: interview
Reasoning: This episode features a conversation between a host and a single
guest, Dr. Priya Nair, discussing her research and expertise on a specific
topic, which matches the structure of an interview episode.
---

The model followed the requested format exactly: "Label:" on its own first
line, "Reasoning:" on the next line. No markdown, no preamble, no code fences.
```

**How did you parse the label out of the response?**

```
Anchor parse (matches Step 3 of the spec):
  1. Split the text on newlines.
  2. Find the line that starts with "label:" (after .strip().lower() and
     stripping any leading markdown like *, `, #). Here: "Label: interview".
  3. Take everything after the first ":" -> " interview", then .strip().lower()
     -> "interview".
  4. Find the line starting with "reasoning:", take the text after the colon
     (plus any continuation lines) as the reasoning string.
  5. Check "interview" is in VALID_LABELS -> it is, so keep it.

For THIS response the fallback paths (whole-text scan for a label keyword,
"unknown" sentinel) were not needed — the primary anchor matched cleanly.
```

**Did any episodes return `"unknown"`? If so, why?**

```
No. This single test parsed cleanly to "interview". A full 20-episode run
(once classify_episode() is implemented) is the real test for whether any
response goes off-format; this section should be updated after that run.
```

**One thing about the output format that surprised you:**

```
The 70B model held the two-line "Label:/Reasoning:" format precisely on the
first try with no fences or preamble — the failure mode I most worried about
when rejecting JSON (markdown wrapping, chatty preamble) didn't appear here.
That said, one clean sample isn't proof; the anchor-based parse and the
"unknown" fallback are still worth keeping for the cases where it doesn't.
```
