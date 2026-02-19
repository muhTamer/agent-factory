# app/orchestration/__init__.py
#
# Lazy imports — modules are imported on first access to avoid
# circular or missing-module errors during incremental development.

from app.orchestration.performance_store import ExecutionRecord, PerformanceStore

__all__ = [
    "ExecutionRecord",
    "PerformanceStore",
]


def __getattr__(name: str):
    """Lazy-import remaining components as they become available."""
    if name in ("SolvabilityEstimator", "SolvabilityResult", "SolvabilityScore"):
        from app.orchestration.solvability_estimator import (  # noqa: F401
            SolvabilityEstimator,
            SolvabilityResult,
            SolvabilityScore,
        )

        return locals()[name]

    if name in ("CompletenessDetector", "CompletenessResult"):
        from app.orchestration.completeness_detector import (  # noqa: F401
            CompletenessDetector,
            CompletenessResult,
        )

        return locals()[name]

    if name in ("AOPCoordinator", "AOPResult", "Subtask"):
        from app.orchestration.aop_coordinator import (  # noqa: F401
            AOPCoordinator,
            AOPResult,
            Subtask,
        )

        return locals()[name]

    raise AttributeError(f"module 'app.orchestration' has no attribute {name!r}")
