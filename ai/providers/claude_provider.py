"""
Anthropic Claude Provider
"""

import os
from typing import Optional, Dict, Any
from ai.providers.base_provider import BaseProvider


class ClaudeProvider(BaseProvider):
    """Anthropic Claude provider via LangChain"""

    def __init__(self, config: Dict[str, Any], logger):
        super().__init__(config, logger)
        self.provider_config = config.get("ai", {}).get("claude", {})
        self.model = self.provider_config.get("model", "claude-sonnet-4-20250514")
        self.api_key = self.provider_config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        self.llm = None
        self._initialize()

    def _initialize(self):
        if not self.api_key:
            self.logger.warning("Anthropic API key not configured")
            return
        try:
            from langchain_anthropic import ChatAnthropic

            self.llm = ChatAnthropic(
                model=self.model,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
            self.logger.info(f"Claude provider initialized with model: {self.model}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Claude: {e}")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[list] = None,
    ) -> str:
        if not self.llm:
            raise RuntimeError("Claude provider not initialized — check API key")

        await self._apply_rate_limit()

        messages = []
        if system_prompt:
            messages.append(("system", system_prompt))
        if context:
            messages.extend(context)
        messages.append(("human", prompt))

        response = await self.llm.ainvoke(messages)
        return response.content

    def generate_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[list] = None,
    ) -> str:
        if not self.llm:
            raise RuntimeError("Claude provider not initialized — check API key")

        self._apply_rate_limit_sync()

        messages = []
        if system_prompt:
            messages.append(("system", system_prompt))
        if context:
            messages.extend(context)
        messages.append(("human", prompt))

        response = self.llm.invoke(messages)
        return response.content

    def get_model_name(self) -> str:
        return self.model

    def is_configured(self) -> bool:
        return bool(self.api_key and self.llm)
