# Evaluation Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:
- **Overall:** What fraction of episodes did we classify correctly?
- **Per-class:** Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What it does
Returns the fraction of predictions that exactly match the ground truth.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`, one per episode. |
| `ground_truth` | `list[str]` | The correct labels, in the same order as `predictions`. |

### Output

| Return value | Type | Description |
|---|---|---|
| accuracy | `float` | A value between 0.0 and 1.0. |

---

### Spec fields — fill these in before writing code

**Formula:**

```
accuracy = (number of positions where prediction == ground_truth)
           / (total number of predictions)

"Correct" = an exact string match between predictions[i] and ground_truth[i]
at the same index. The two lists are parallel (same length, same order), so we
compare position by position. We divide by the total count of predictions
(== len(ground_truth)), NOT by the number of correct ones. A label of
"unknown" never equals a real ground-truth label, so it simply counts as wrong.
```

---

**Step-by-step logic:**

```
1. If predictions is empty (len == 0), return 0.0 immediately to avoid
   dividing by zero (see edge case below).
2. Pair up predictions and ground_truth by index (zip).
3. Count how many pairs have prediction == ground_truth.
4. Divide that count by len(predictions) to get a float in [0.0, 1.0].
5. Return the float.
```

---

**Edge case — what if both lists are empty?**

```
Return 0.0. With no predictions there is nothing correct, and dividing by
len(predictions) == 0 would raise ZeroDivisionError. 0.0 is a safe, in-range
sentinel that the report formatter ({accuracy:.1%}) can display without
crashing. (Assumes the two lists are always equal length, which run_evaluation
guarantees — they are built from the same results list.)
```

---

**Worked example:**

```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

Compare index by index:
  i=0: interview == interview  -> correct
  i=1: solo      == solo       -> correct
  i=2: panel     != solo       -> wrong
  i=3: interview != narrative  -> wrong

correct = 2, total = 4
accuracy = 2 / 4 = 0.5

compute_accuracy() returns 0.5
```

---

## compute_per_class_accuracy(predictions, ground_truth)

### What it does
Returns accuracy broken down by each label. For each label in `VALID_LABELS`,
reports how many episodes with that ground-truth label were classified correctly.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`. |
| `ground_truth` | `list[str]` | Correct labels, in the same order. |

### Output

A `dict` keyed by label. Each value is a dict with three keys:

```python
{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}
```

---

### Spec fields — fill these in before writing code

**What does "correct" mean for a given class?**

```
An episode counts as correct FOR class C when its ground_truth is C AND the
prediction also equals C. For the "interview" class: ground_truth[i] ==
"interview" and predictions[i] == "interview". Because we only count an episode
under its ground-truth class, "correct for C" is just (truth == C and
pred == C) — which, given truth == C, is the same as pred == truth.
```

---

**What does "total" mean for a given class?**

```
"total" for class C = the number of episodes whose GROUND_TRUTH label is C —
i.e. how many true C's exist in the test set. It is NOT the number of times C
was predicted, and NOT the overall number of predictions. This makes per-class
accuracy = "of the real C episodes, what fraction did we get right?" (recall
for that class). Consequence: the per-class totals sum to len(ground_truth),
and predictions that are "unknown" or a wrong class lower the accuracy of the
TRUE class's bucket, not of the class that was wrongly predicted.
```

---

**Step-by-step logic:**

```
[blank — describe the steps your code will take.
 1. Initialize ...
 2. Loop over ...
 3. For each pair (predicted, truth) ...
 4. After the loop ...
 5. Return ...]
```

---

**Edge case — what if a class has no examples in ground_truth (total == 0)?**

```
[blank — what should accuracy be set to? Why?
 Hint: look at the docstring in evaluate.py.]
```

---

**Worked example:**

```
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

[blank — fill in the per-class results table below]

label       correct  total  accuracy
----------  -------  -----  --------
interview   [blank]  [blank]  [blank]
solo        [blank]  [blank]  [blank]
panel       [blank]  [blank]  [blank]
narrative   [blank]  [blank]  [blank]
```

---

## Reflection questions (discuss at the checkpoint)

1. Your overall accuracy might be decent even if one class has very low accuracy.
   Why is per-class accuracy a more informative metric than overall accuracy alone?

2. If `panel` episodes consistently get misclassified as `interview`, what does
   that tell you about your training labels or your prompt?

3. You labeled 20 training episodes and evaluated on 20 test episodes (5 per class).
   How might the evaluation results change if you had labeled 100 training episodes?
   What if you had 200 test episodes?
