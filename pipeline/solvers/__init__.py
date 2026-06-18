"""Per-category solvers. Each module exposes solve(prompt[, vocab]) -> (answer, cot)."""
from .common import verify
from . import numeral, unit_conversion, gravity, cipher
# simple-signature solvers (cipher needs vocab, handled by caller)
SOLVERS = {
    'numeral': numeral.solve,
    'unit_conversion': unit_conversion.solve,
    'gravity': gravity.solve,
}
