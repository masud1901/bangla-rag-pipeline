import openai
import logging
from typing import List, Dict, Any, Tuple
from src.core.config import settings

logger = logging.getLogger(__name__)

class OpenAILLMService:
    """
    Service for generating answers using OpenAI's models (e.g., GPT-3.5-Turbo).
    """
    def __init__(self):
        """Initialize the OpenAI client."""
        try:
            # Create OpenAI client with minimal configuration to avoid httpx issues
            import os
            api_key = settings.openai_api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found")
            
            self.client = openai.OpenAI(
                api_key=api_key,
                timeout=30.0
            )
            self.model_name = "gpt-3.5-turbo"
            logger.info(f"Initialized OpenAI client with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def _create_system_prompt(self) -> str:
        """Creates the system prompt that defines the AI's persona and task."""
        return """You are a knowledgeable and precise tutor for Bengali literature. 
Your task is to answer questions based EXACTLY on the provided textbook content.

**Instructions:**
1.  Read the provided context carefully and identify the most relevant information.
2.  Answer the question using ONLY the information from the provided context.
3.  If the exact answer is in the context, provide it clearly and accurately.
4.  If the context is related but doesn't contain the exact answer, say: "Based on the provided context, I cannot find the specific answer to this question."
5.  Do NOT make up information or use external knowledge.
6.  Be precise and factual - avoid speculation.
7.  If the question is in Bengali, respond in Bengali. If in English, respond in English.
8.  Keep answers concise but complete.
9.  If you find multiple relevant pieces of information, synthesize them clearly.
"""

    def _create_user_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Creates the user-facing prompt with the context and question."""
        if not context_chunks:
            return f"User question: {query}\n\nAnswer directly from your own knowledge, as no specific context was found."

        context_str = "\n\n---\n\n".join([chunk['text'] for chunk in context_chunks])
        
        return f"""Textbook Content:
{context_str}

Question: {query}

Instructions: Answer the question using ONLY the information from the textbook content above. Be precise and accurate. If the exact answer is not in the provided content, say so clearly."""

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """
        Generates an answer using the provided query and context.
        
        Args:
            query: The user's question.
            context_chunks: A list of relevant document chunks.

        Returns:
            A tuple containing the generated answer and usage metadata.
        """
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(query, context_chunks)
        
        try:
            logger.info(f"Sending request to OpenAI with model {self.model_name}")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500,
            )

            answer = response.choices[0].message.content
            usage_info = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            logger.info(f"Successfully received answer from OpenAI. Usage: {usage_info['total_tokens']} tokens.")
            return answer, usage_info

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return "There was an error generating an answer.", {"error": str(e)}

    def get_model_info(self) -> Dict[str, str]:
        """Returns information about the LLM."""
        return {
            "provider": "OpenAI",
            "model_name": self.model_name,
        } 