"""
Service to transform/optimize input queries
"""

import re

from django.conf import settings

from integreat_chat.chatanswers.services.litellm import LiteLLMClient

from ..static.prompts import Prompts


class QueryTransformer:
    """
    Class that enables transforming complicated queries based on different criteria
    """

    def __init__(self, original_query):
        self.original_query = original_query
        self.modified_query = ""
        self.llm_api = LiteLLMClient(
            Prompts.CHECK_SYSTEM_PROMPT,
            settings.RAG_QUERY_OPTIMIZATION_MODEL
        )
        self.length_threshold_chars = 150

    def punctuation_thresh_exceeded(self) -> bool:
        """
        Identify punctuations(period, comma, question_mark) to identify complexity
        """
        max_period = 1
        max_question_mark = 1
        max_comma = 2

        punctuation_patterns = {
            "period": r"[.。]",
            "comma": r"[,،，、]",
            "question_mark": r"[?؟？]",
        }
        counts = {
            punct: len(re.findall(pattern, self.original_query))
            for punct, pattern in punctuation_patterns.items()
        }

        if (
            counts["period"] > max_period
            or counts["question_mark"] > max_question_mark
            or counts["comma"] > max_comma
        ):
            return True
        return False

    def length_thresh_exceeded(self):
        """
        Check if the query exceeds a certain length threshold
        """
        return len(self.original_query) > self.length_threshold_chars

    def is_transformation_required(self) -> bool:
        """
        Check if the query requires transformation
        """
        if self.punctuation_thresh_exceeded() or self.length_thresh_exceeded():
            return True
        return False

    def transform_query(self):
        """
        Optimize the user query for document retrieval
        """
        self.modified_query = self.llm_api.simple_prompt(
            Prompts.OPTIMIZE_MESSAGE.format(self.original_query)
        )
        return {
            "original_query": self.original_query,
            "modified_query": self.modified_query,
        }
