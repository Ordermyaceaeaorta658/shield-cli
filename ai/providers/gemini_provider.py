"""
Google Gemini Provider
"""

import os
from typing import Optional, Dict, Any
from ai.providers.base_provider import BaseProvider


class GeminiProvider(BaseProvider):
    """Google Gemini provider via LangChain"""

    def __init__(self, config: Dict[str, Any], logger):
        super().__init__(config, logger)
        self.provider_config = config.get("ai", {}).get("gemini", {})
        self.model = self.provider_config.get("model", "gemini-2.5-pro")
        self.api_key = self.provider_config.get("api_key") or os.getenv("GOOGLE_API_KEY")
        self.llm = None
        self._initialize()

    def _initialize(self):
        if not self.api_key:
            self.logger.warning("Google API key not configured")
            return
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self.llm = ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=self.api_key,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
            self.logger.info(f"Gemini provider initialized with model: {self.model}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini: {e}")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[list] = None,
    ) -> str:
        if not self.llm:
            raise RuntimeError("Gemini provider not initialized — check API key")

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
            raise RuntimeError("Gemini provider not initialized — check API key")

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
