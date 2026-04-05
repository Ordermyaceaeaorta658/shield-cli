"""
OpenAI GPT-4 Provider
"""

import os
from typing import Optional, Dict, Any
from ai.providers.base_provider import BaseProvider


class OpenAIProvider(BaseProvider):
    """OpenAI GPT-4 provider via LangChain"""

    def __init__(self, config: Dict[str, Any], logger):
        super().__init__(config, logger)
        self.provider_config = config.get("ai", {}).get("openai", {})
        self.model = self.provider_config.get("model", "gpt-4o")
        self.api_key = self.provider_config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.llm = None
        self._initialize()

    def _initialize(self):
        """Initialize the OpenAI LLM"""
        if not self.api_key:
            self.logger.warning("OpenAI API key not configured")
            return
        try:
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(
                model=self.model,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
            self.logger.info(f"OpenAI provider initialized with model: {self.model}")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI: {e}")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[list] = None,
    ) -> str:
        if not self.llm:
            raise RuntimeError("OpenAI provider not initialized — check API key")

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
            raise RuntimeError("OpenAI provider not initialized — check API key")

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
