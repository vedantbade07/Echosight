"""
Hotkey and voice-command routing: lets the user trigger scene narration
or document reading on demand, hands-free. Original orchestration logic
for EchoSight.
"""

from typing import Callable, Dict


class CommandRouter:
    def __init__(self):
        self._handlers: Dict[str, Callable[[], None]] = {}

    def register(self, command: str, handler: Callable[[], None]) -> None:
        self._handlers[command.lower()] = handler

    def dispatch(self, spoken_text: str) -> bool:
        command = spoken_text.strip().lower()
        for key, handler in self._handlers.items():
            if key in command:
                handler()
                return True
        return False
