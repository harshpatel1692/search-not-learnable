"""Shared: grader-exact verify()."""
import re, math
def verify(stored, predicted):
    stored, predicted = str(stored).strip(), str(predicted).strip()
    if re.fullmatch(r'[01]+', stored): return predicted.lower()==stored.lower()
    try: return math.isclose(float(stored), float(predicted), rel_tol=1e-2, abs_tol=1e-5)
    except Exception: return predicted.lower()==stored.lower()
