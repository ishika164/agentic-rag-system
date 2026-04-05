from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class Turn:
    role: str
    content: str


class ConversationMemory:
    def __init__(self, window: int = 3) -> None:
        self._turns: Deque[Turn] = deque(maxlen=window * 2)

    def add_turn(self, role: str, content: str) -> None:
        self._turns.append(Turn(role=role, content=content))

    def add_exchange(self, question: str, answer: str) -> None:
        self.add_turn("Human", question)
        self.add_turn("Assistant", answer)

    def format(self) -> str:
        if not self._turns:
            return ""
        lines = []
        for t in self._turns:
            lines.append(f"{t.role}: {t.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)