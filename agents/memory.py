"""
Conversational State, Short-Term Buffer, and Scratchpad Memory for Agents.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agents.base import AgentMessage, AgentStep


class AgentMemory(BaseModel):
    """
    Dual-memory container:
    1. Short-term conversational message history (FIFO buffer).
    2. Dynamic execution scratchpad (stores intermediate steps and facts).
    """

    max_history: int = Field(default=20, ge=1)
    messages: List[AgentMessage] = Field(default_factory=list)
    scratchpad: Dict[str, Any] = Field(default_factory=dict)
    steps: List[AgentStep] = Field(default_factory=list)

    def add_message(self, role: str, content: str, name: Optional[str] = None) -> None:
        """Append message to short-term conversational buffer."""
        msg = AgentMessage(role=role, content=content, name=name, timestamp=time.time())
        self.messages.append(msg)
        # Truncate if exceeds max history
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history :]

    def add_step(self, step: AgentStep) -> None:
        """Record an executed ReAct reasoning step."""
        self.steps.append(step)

    def set_scratchpad(self, key: str, value: Any) -> None:
        """Store key-value fact into working scratchpad."""
        self.scratchpad[key] = value

    def get_scratchpad(self, key: str, default: Any = None) -> Any:
        """Retrieve stored fact from working scratchpad."""
        return self.scratchpad.get(key, default)

    def get_recent_context_str(self) -> str:
        """Render recent conversational turns as a string prompt context."""
        lines = []
        for m in self.messages[-6:]:
            prefix = f"{m.name} ({m.role})" if m.name else m.role.capitalize()
            lines.append(f"{prefix}: {m.content}")
        return "\n".join(lines)

    def clear_scratchpad(self) -> None:
        """Clear intermediate step scratchpad for next turn."""
        self.scratchpad.clear()
        self.steps.clear()
