import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ActionNormalizer:
    action_min: Optional[np.ndarray] = None
    action_max: Optional[np.ndarray] = None
    range_eps: float = 1e-7

    @property
    def fitted(self) -> bool:
        return self.action_min is not None and self.action_max is not None

    def fit(self, actions: np.ndarray) -> "ActionNormalizer":
        actions = np.asarray(actions)
        if actions.ndim < 2:
            raise ValueError(f"actions must have shape (N, D) or (.., D), got {actions.shape}")
        flat = actions.reshape(-1, actions.shape[-1])
        action_min = flat.min(axis=0)
        action_max = flat.max(axis=0)

        small_range = (action_max - action_min) < self.range_eps
        action_max = action_max.copy()
        action_max[small_range] = action_min[small_range] + 1.0

        self.action_min = action_min
        self.action_max = action_max
        return self

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("ActionNormalizer is not fitted.")
        actions = np.asarray(actions, dtype=np.float32)
        denom = (self.action_max - self.action_min).astype(np.float32)
        return (actions - self.action_min.astype(np.float32)) / denom * 2.0 - 1.0

    def unnormalize(self, actions: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("ActionNormalizer is not fitted.")
        actions = np.asarray(actions, dtype=np.float32)
        scale = (self.action_max - self.action_min).astype(np.float32)
        return (actions + 1.0) / 2.0 * scale + self.action_min.astype(np.float32)

    def to_state_dict(self) -> Dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("ActionNormalizer is not fitted.")
        return {
            "action_min": self.action_min,
            "action_max": self.action_max,
            "range_eps": self.range_eps,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "ActionNormalizer":
        normalizer = cls()
        normalizer.action_min = state["action_min"]
        normalizer.action_max = state["action_max"]
        normalizer.range_eps = state.get("range_eps", 1e-7)
        return normalizer

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.to_state_dict(), f)

    @classmethod
    def load(cls, path: str) -> "ActionNormalizer":
        with open(path, "rb") as f:
            state = pickle.load(f)
        return cls.from_state_dict(state)