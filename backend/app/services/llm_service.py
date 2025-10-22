"""
LLM Service Integration Layer

This service provides a unified interface for multiple LLM providers:
- Google Gemini
- OpenAI
- Anthropic Claude

It handles code generation, validation, and fallback mechanisms.
"""

import logging
import os
from typing import Dict, Any, Optional, List
from enum import Enum
import json

# Import LLM clients (optional imports)
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class LLMService:
    """
    Unified LLM service for code generation and analysis.
    
    Supports multiple providers with automatic fallback.
    """
    
    def __init__(self, primary_provider: LLMProvider = LLMProvider.GEMINI):
        """
        Initialize LLM service with specified primary provider.
        
        Args:
            primary_provider: Primary LLM provider to use
        """
        self.primary_provider = primary_provider
        self.logger = logging.getLogger("llm_service")
        
        # Initialize clients
        self._init_clients()
        
    def _init_clients(self):
        """Initialize LLM clients for all available providers"""
        self.clients = {}
        
        # Initialize Gemini
        if genai and settings.google_api_key:
            try:
                genai.configure(api_key=settings.google_api_key)
                self.clients[LLMProvider.GEMINI] = genai.GenerativeModel('gemini-pro')
                self.logger.info("Gemini client initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Gemini: {e}")
        
        # Initialize OpenAI
        if OpenAI and settings.openai_api_key:
            try:
                self.clients[LLMProvider.OPENAI] = OpenAI(api_key=settings.openai_api_key)
                self.logger.info("OpenAI client initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI: {e}")
        
        # Initialize Anthropic
        if Anthropic and settings.anthropic_api_key:
            try:
                self.clients[LLMProvider.ANTHROPIC] = Anthropic(api_key=settings.anthropic_api_key)
                self.logger.info("Anthropic client initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Anthropic: {e}")
    
    async def generate_code(
        self,
        prompt: str,
        context: Dict[str, Any],
        code_type: str = "data_cleaning",
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Generate Python code using LLM based on analysis context.
        
        Args:
            prompt: Base prompt for code generation
            context: Analysis context from Layer 1
            code_type: Type of code to generate
            max_retries: Maximum retry attempts
            
        Returns:
            Dictionary containing generated code and metadata
        """
        # Build comprehensive prompt
        full_prompt = self._build_code_generation_prompt(prompt, context, code_type)
        
        # Try primary provider first
        for attempt in range(max_retries):
            try:
                result = await self._generate_with_provider(
                    self.primary_provider,
                    full_prompt
                )
                
                if result and result.get("code"):
                    self.logger.info(f"Code generated successfully with {self.primary_provider}")
                    return result
                    
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt + 1} failed with {self.primary_provider}: {e}"
                )
                
                # Try fallback providers
                if attempt == max_retries - 1:
                    result = await self._try_fallback_providers(full_prompt)
                    if result:
                        return result
        
        raise Exception("Failed to generate code with all providers")
    
    async def _generate_with_provider(
        self,
        provider: LLMProvider,
        prompt: str
    ) -> Optional[Dict[str, Any]]:
        """Generate code with specific provider"""
        if provider not in self.clients:
            return None
        
        try:
            if provider == LLMProvider.GEMINI:
                return await self._generate_with_gemini(prompt)
            elif provider == LLMProvider.OPENAI:
                return await self._generate_with_openai(prompt)
            elif provider == LLMProvider.ANTHROPIC:
                return await self._generate_with_anthropic(prompt)
        except Exception as e:
            self.logger.error(f"Error with {provider}: {e}")
            return None
    
    async def _generate_with_gemini(self, prompt: str) -> Dict[str, Any]:
        """Generate code using Google Gemini"""
        model = self.clients[LLMProvider.GEMINI]
        response = model.generate_content(prompt)
        
        code = self._extract_code_from_response(response.text)
        
        return {
            "code": code,
            "provider": "gemini",
            "raw_response": response.text,
            "metadata": {
                "model": "gemini-pro",
                "tokens": len(response.text.split())
            }
        }
    
    async def _generate_with_openai(self, prompt: str) -> Dict[str, Any]:
        """Generate code using OpenAI"""
        client = self.clients[LLMProvider.OPENAI]
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert Python programmer specializing in data science and ML."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        code = self._extract_code_from_response(response.choices[0].message.content)
        
        return {
            "code": code,
            "provider": "openai",
            "raw_response": response.choices[0].message.content,
            "metadata": {
                "model": "gpt-4",
                "tokens": response.usage.total_tokens
            }
        }
    
    async def _generate_with_anthropic(self, prompt: str) -> Dict[str, Any]:
        """Generate code using Anthropic Claude"""
        client = self.clients[LLMProvider.ANTHROPIC]
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        code = self._extract_code_from_response(response.content[0].text)
        
        return {
            "code": code,
            "provider": "anthropic",
            "raw_response": response.content[0].text,
            "metadata": {
                "model": "claude-3-5-sonnet-20241022",
                "tokens": response.usage.input_tokens + response.usage.output_tokens
            }
        }
    
    async def _try_fallback_providers(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Try all fallback providers"""
        for provider in LLMProvider:
            if provider != self.primary_provider and provider in self.clients:
                self.logger.info(f"Trying fallback provider: {provider}")
                result = await self._generate_with_provider(provider, prompt)
                if result:
                    return result
        return None
    
    def _build_code_generation_prompt(
        self,
        base_prompt: str,
        context: Dict[str, Any],
        code_type: str
    ) -> str:
        """Build comprehensive prompt for code generation"""
        prompt = f"""# Task: Generate Python Code for {code_type}

## Base Requirements:
{base_prompt}

## Analysis Context (Layer 1 Results):
{json.dumps(context, indent=2)}

## Code Generation Guidelines:
1. Generate clean, well-documented Python code
2. Include proper error handling
3. Use pandas, numpy, scikit-learn as needed
4. Add detailed comments explaining each step
5. Follow PEP 8 style guidelines
6. Ensure code is self-contained and executable
7. Return ONLY the Python code, wrapped in ```python``` code blocks

## Output Format:
```python
# Your generated code here
```

Generate the code now:
"""
        return prompt
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response"""
        # Try to find code blocks
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        # If no code blocks, return entire response
        return response.strip()
    
    async def generate_explanation(
        self,
        code: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Generate educational explanation for generated code.
        
        Args:
            code: Generated code to explain
            context: Context about the data and task
            
        Returns:
            Educational explanation in plain English
        """
        prompt = f"""Explain the following Python code in simple, educational terms for a non-technical user:

## Code:
```python
{code}
```

## Context:
{json.dumps(context, indent=2)}

Provide a clear, step-by-step explanation of:
1. What the code does
2. Why each step is necessary
3. What data science concepts are being applied
4. Expected outcomes

Keep the explanation accessible and educational.
"""
        
        try:
            result = await self._generate_with_provider(self.primary_provider, prompt)
            if result:
                return result.get("raw_response", "")
        except Exception as e:
            self.logger.error(f"Failed to generate explanation: {e}")
        
        return "Explanation generation failed."


# Global LLM service instance
_llm_service = None


def get_llm_service(provider: LLMProvider = LLMProvider.GEMINI) -> LLMService:
    """Get or create global LLM service instance"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService(provider)
    return _llm_service

