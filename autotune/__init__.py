"""Generated Transformer-program search with constraint-aware Optuna TPE."""

from .engine import (
    SearchBudget,
    SearchEngine,
    SearchPlan,
    SearchRequest,
    SearchResult,
)
from .evaluator import (
    PROMOTION_BLOCK_COUNT,
    PROMOTION_BLOCK_WIN_RATIO,
    PROMOTION_REQUIRED_WINS,
    RESIDENT_PROTOCOLS,
    STREAMED_PROTOCOLS,
    ConstraintVector,
    EvaluationScope,
    Evaluator,
    Fidelity,
    FidelityProtocol,
    PairedMeasurement,
    TrialMeasurement,
)
from .optuna_store import CompletedTrial, OptunaBackend
from .space import (
    BranchSpace,
    ParameterDomain,
    ProgramSearchSpace,
    SearchContext,
    StructureSpec,
)
from .storage import SearchStorage, StudyIdentity

__all__ = [
    "PROMOTION_BLOCK_COUNT",
    "PROMOTION_BLOCK_WIN_RATIO",
    "PROMOTION_REQUIRED_WINS",
    "RESIDENT_PROTOCOLS",
    "STREAMED_PROTOCOLS",
    "BranchSpace",
    "CompletedTrial",
    "ConstraintVector",
    "EvaluationScope",
    "Evaluator",
    "Fidelity",
    "FidelityProtocol",
    "OptunaBackend",
    "PairedMeasurement",
    "ParameterDomain",
    "ProgramSearchSpace",
    "SearchBudget",
    "SearchContext",
    "SearchEngine",
    "SearchPlan",
    "SearchRequest",
    "SearchResult",
    "SearchStorage",
    "StructureSpec",
    "StudyIdentity",
    "TrialMeasurement",
]
