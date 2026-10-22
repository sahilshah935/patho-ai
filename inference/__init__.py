"""pathology_ai.inference — clinical interpretation and explanation."""

from pathology_ai.inference.interpreter import Interpreter, InterpretationResult
from pathology_ai.inference.explainer import explain_and_summarise, generate_disclaimer
from pathology_ai.inference.chatbot import PathologyChatbot

__all__ = [
    "Interpreter",
    "InterpretationResult",
    "explain_and_summarise",
    "generate_disclaimer",
    "PathologyChatbot",
]
