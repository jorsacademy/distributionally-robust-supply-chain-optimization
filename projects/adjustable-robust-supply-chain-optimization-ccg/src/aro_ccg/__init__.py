"""Adjustable robust supply-chain optimization with C&CG."""

from .baselines import BenchmarkComparison, compare_methods
from .ccg import CCGResult, solve_ccg
from .data import SupplyChainInstance, demo_instance
from .experiments import SensitivityRecord, run_sensitivity

__all__ = [
    "BenchmarkComparison",
    "CCGResult",
    "SensitivityRecord",
    "SupplyChainInstance",
    "compare_methods",
    "demo_instance",
    "run_sensitivity",
    "solve_ccg",
]
__version__ = "0.2.0"
