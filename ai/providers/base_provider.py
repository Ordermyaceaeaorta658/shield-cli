"""
Base Provider Interface for AI Models
Defines the common interface all AI providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import time
import asyncio


class BaseProvider(ABC):
    """Abstract base class for AI providers"""

    def __init__(self, config: Dict[str, Any], logger):
        self.config = config
        self.logger = logger

        # Rate limiting
        ai_config = config.get("ai", {})
        self.rate_limit = ai_config.get("rate_limit", 2)
        self._min_interval = 60.0 / self.rate_limit if self.rate_limit > 0 else 0
        self._last_request = 0.0

        # General settings
        self.temperature = ai_config.get("temperature", 0.2)
        self.max_tokens = ai_config.get("max_tokens", 8000)
        self.timeout = ai_config.get("timeout", 60)

    @abstractmethod
    def _initialize(self):
        """Initialize the provider backend"""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[list] = None,
    ) -> str:
        """Generate a response asynchronously"""
        pass

    @abstractmethod
    def generate_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[list] = None,
    ) -> str:
        """Generate a response synchronously"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier"""
        pass

    async def generate_with_reasoning(
        self,
        prompt: str,
        system_prompt: str,
        task_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate response with explicit reasoning steps.
        Providers can override this for native reasoning support.
        """
        full_prompt = (
            f"{prompt}\n\n"
            "Please think through this step-by-step:\n"
            "1. First, explain your REASONING\n"
            "2. Then provide your RESPONSE\n\n"
            "Format:\nREASONING: <your analysis>\nRESPONSE: <your answer>"
        )

        if task_context:
            full_prompt = f"Context:\n{task_context}\n\n{full_prompt}"

        response = await self.generate(full_prompt, system_prompt)

        # Parse reasoning and response
        reasoning = response
        final_response = response

        if "REASONING:" in response and "RESPONSE:" in response:
            parts = response.split("RESPONSE:", 1)
            reasoning = parts[0].replace("REASONING:", "").strip()
            final_response = parts[1].strip() if len(parts) > 1 else response

        return {"response": final_response, "reasoning": reasoning}

    async def _apply_rate_limit(self):
        """Apply rate limiting between API calls"""
        if self._min_interval > 0:
            elapsed = time.time() - self._last_request
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
        self._last_request = time.time()

    def _apply_rate_limit_sync(self):
        """Apply rate limiting (synchronous version)"""
        if self._min_interval > 0:
            elapsed = time.time() - self._last_request
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()
