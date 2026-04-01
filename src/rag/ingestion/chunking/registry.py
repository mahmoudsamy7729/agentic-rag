from __future__ import annotations

from src.rag.ingestion.chunking.base import ChunkingStrategy


class ChunkingStrategyRegistry:
    def __init__(self, strategies: list[ChunkingStrategy]) -> None:
        self._strategies = {strategy.name: strategy for strategy in strategies}
        if not self._strategies:
            raise ValueError("At least one chunking strategy must be registered.")

    def resolve(self, name: str) -> ChunkingStrategy:
        strategy = self._strategies.get(name)
        if strategy is None:
            available = ", ".join(sorted(self._strategies))
            raise ValueError(
                f"Unsupported chunking_strategy '{name}'. Available: {available}."
            )
        return strategy

    def names(self) -> list[str]:
        return sorted(self._strategies)
