"""Lightweight src package initializer.

Avoids importing heavy modules at import time so subpackages (e.g., platform.adapters)
can be used without full platform dependencies.
"""

try:
    from .registry import ModelRegistry  # type: ignore
    from .model_logistic import train_and_eval as logistic_train_eval  # type: ignore

    try:
        ModelRegistry.register("logistic", logistic_train_eval)  # type: ignore[attr-defined]
    except Exception:
        pass
except Exception:
    # Safe no-op if optional components are unavailable in this context
    pass
