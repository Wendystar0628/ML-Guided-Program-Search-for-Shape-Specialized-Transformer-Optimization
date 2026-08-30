"""Generated Transformer-program search with constraint-aware Optuna TPE."""

from .evaluation import (
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
from .promotion import (
    PROMOTION_BASE_RATIO,
    PROMOTION_BASE_WINS,
    PROMOTION_MAX_BLOCKS,
    PROMOTION_STAGES,
    PromotionDecision,
    promotion_decision,
)
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
    "PROMOTION_BASE_RATIO",
    "PROMOTION_BASE_WINS",
    "PROMOTION_MAX_BLOCKS",
    "PROMOTION_STAGES",
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
    "PromotionDecision",
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
    "promotion_decision",
]
