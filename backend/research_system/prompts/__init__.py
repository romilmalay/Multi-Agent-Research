"""Prompt templates, one YAML file per LLM agent."""

from research_system.prompts.registry import Prompt, load

__all__ = ["Prompt", "load"]
