"""RL reward for the Tinker campaign. FULLY LOCAL — no Tinker dependency.

reward(sample_text, gold) -> 1.0 / 0.0

Two deliberate differences from the official grader (both make the reward
STRICTER, never looser — anti-reward-hacking):
  1. boxed-only extraction: the grader falls back to "last number in the text"
     when no \\boxed{} exists; rewarding that would teach number-spam.
  2. None on no-box: unboxed rollouts score 0 even if a correct number appears.

verify() itself is VERBATIM the competition metric (binary exact -> float
rel_tol 1e-2 -> case-insensitive string), so anything we reward, the grader
also accepts.

Self-test: python3 tinker/reward.py
"""
import math
import re


def extract_boxed_strict(text):
    """Last NON-EMPTY \\boxed{...} via the grader's brace-balanced scan.
    Returns None when no boxed answer exists (NO fallback — intentional)."""
    if text is None:
        return None
    starts = list(re.finditer(r"\\boxed\{", text))
    matches = []
    for i, m in enumerate(starts):
        s = m.end()
        e = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        seg = text[s:e]
        lb = seg.rfind("}")
        matches.append(seg[:lb] if lb != -1 else seg)
    non_empty = [m.strip() for m in matches if m.strip()]
    return non_empty[-1] if non_empty else None


def verify(stored_answer, predicted):
    """VERBATIM competition metric (do not edit)."""
    stored = str(stored_answer).strip()
    pred = str(predicted).strip()
    if re.fullmatch(r"[01]+", stored):
        return pred.lower() == stored.lower()
    try:
        return math.isclose(float(stored), float(pred), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return pred.lower() == stored.lower()


def reward(sample_text, gold, truncated=False):
    """Binary verifiable reward. truncated=True (hit max_tokens) always 0 —
    budget discipline is part of the trained skill."""
    if truncated:
        return 0.0
    pred = extract_boxed_strict(sample_text)
    if pred is None:
        return 0.0
    return 1.0 if verify(gold, pred) else 0.0


if __name__ == "__main__":
    cases = [
        # (sample, gold, truncated, expected)
        ("... \\boxed{10110001}", "10110001", False, 1.0),
        ("... \\boxed{10110001}", "10110001", True, 0.0),   # truncation -> 0
        ("answer is 10110001 no box", "10110001", False, 0.0),  # strict: no fallback
        ("\\boxed{} then \\boxed{42}", "42.0", False, 1.0),  # float tol + last non-empty
        ("\\boxed{ALICE discovers}", "alice discovers", False, 1.0),  # case-insens
        ("\\boxed{}}^}", "}}^", False, 1.0),                 # cryptarithm literal braces
        ("\\boxed{0110}", "110", False, 0.0),                # binary: leading zeros matter
        ("\\boxed{3.1416}", "3.14", False, 1.0),             # rel_tol 1e-2
        ("nothing here", "x", False, 0.0),
    ]
    for s, g, t, want in cases:
        got = reward(s, g, truncated=t)
        assert got == want, (s, g, t, got, want)
    print(f"reward.py self-test: {len(cases)}/{len(cases)} pass")
