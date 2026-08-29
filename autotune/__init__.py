"""Generated Transformer-program search with constraint-aware Optuna TPE."""

from .evaluation import (
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
from .optimization_loop import (
    OptimizationIteration,
    OptimizationLoop,
    OptimizationLoopPolicy,
    OptimizationResult,
)
from .optuna_backend import CompletedTrial, OptunaBackend
from .search_engine import (
    SearchBudget,
    SearchEngine,
    SearchPlan,
    SearchRequest,
    SearchResult,
)
from .search_space import (
    BranchSpace,
    ParameterDomain,
    ProgramSearchSpace,
    SearchContext,
    StructureSpec,
)
from .study_storage import SearchStorage, StudyIdentity

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
    "OptimizationIteration",
    "OptimizationLoop",
    "OptimizationLoopPolicy",
    "OptimizationResult",
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
