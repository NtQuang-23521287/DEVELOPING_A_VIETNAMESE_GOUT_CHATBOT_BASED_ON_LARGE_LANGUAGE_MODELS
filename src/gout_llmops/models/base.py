from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModelBackend(ABC):
    @abstractmethod
    def generate(self, messages: list[dict[str, str]], max_input_tokens: int) -> dict[str, Any]:
        raise NotImplementedError
