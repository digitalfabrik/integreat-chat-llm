"""
A service to detect languages and translate messages
"""

import logging
import hashlib

# pylint: disable=no-name-in-module
from transformers import pipeline

from django.conf import settings
from django.core.cache import cache
from integreat_chat.chatanswers.services.litellm import LiteLLMClient

from ..static.prompts import Prompts
from ..static.language_code_map import LANGUAGE_MAP

LOGGER = logging.getLogger("django")


class LanguageService:
    """
    Service class that enables searching for Integreat content
    """
    def __init__(self):
        """
        
        """
        self.llm_api = LiteLLMClient(Prompts.SYSTEM_PROMPT, settings.LANGUAGE_CLASSIFICATIONH_MODEL)

    def classify_language(self, estimated_lang, message):
        """
        Check if a message fits the estimated language.
        Return another language tag, if it does not fit.
        """
        LOGGER.debug("Detecting message language")
        answer = self.llm_api.simple_prompt(Prompts.LANGUAGE_CLASSIFICATION.format(message))
        LOGGER.debug("Finished message language detection: %s", answer)
        if answer.startswith(estimated_lang):
            return estimated_lang
        return answer.split("-")[0]

    def translate_message(self, source_language, target_language, message):
        """
        Translate a message from source to target language
        """
        cache_id = hashlib.sha256(
            f"{source_language}-{target_language}-{message}".encode('utf-8')
        ).hexdigest()
        if source_language == target_language:
            LOGGER.debug("Skipping translation from %s to %s", source_language, target_language)
            return message
        if translated_message := cache.get(cache_id):
            return translated_message
        LOGGER.debug("Starting translation from %s to %s", source_language, target_language)
        pipe = pipeline("translation", model=settings.TRANSLATION_MODEL)
        LOGGER.debug("Finished translation from %s to %s", source_language, target_language)
        translated_message = " ".join([
            result["translation_text"] for result in pipe(
                self.split_text(message),
                tgt_lang=LANGUAGE_MAP[target_language],
                src_lang=LANGUAGE_MAP[source_language]
            )
        ])
        cache.set(cache_id, translated_message)
        return translated_message

    def opportunistic_translate(self, expected_language, message):
        """
        Translate if detected language does not fit the expected language
        """
        classified_language = self.classify_language(
            expected_language,
            message
        )
        return (
            message if classified_language == expected_language
            else self.translate_message(
                classified_language,
                expected_language,
                message
            )
        )

    def split_text(self, text, max_length=200):
        """
        Chunk text at the end of sentences into max 500 char chunks
        """
        sentences = text.split('.')
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if not sentence.strip():
                continue
            sentence = sentence.strip() + "."
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        LOGGER.debug("Translation chunks:\n - %s", "\n - ".join(chunks))
        return chunks
