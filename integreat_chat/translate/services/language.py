"""
A service to detect languages and translate messages
"""

import logging
import hashlib
import re
import spacy

# pylint: disable=no-name-in-module
from transformers import pipeline

from django.conf import settings
from django.core.cache import cache
from integreat_chat.chatanswers.services.llmapi import (
    LlmApiClient, LlmMessage, LlmPrompt, LlmResponse
)

from ..static.prompts import Prompts
from ..static.language_code_map import LANGUAGE_MAP
from ..static.language_classification_map import LANGUAGE_CLASSIFICATION_MAP

LOGGER = logging.getLogger("django")


class LanguageService:
    """
    Service class that enables searching for Integreat content
    """

    def __init__(self):
        """ """
        self.llm_api = LlmApiClient()

    def parse_language(self, response: dict) -> str:
        """
        Parse String with language classification received from model
        """
        classfied_language = response["bcp47-tag"]
        stripped_language = classfied_language.split("-")[0]
        if stripped_language in LANGUAGE_CLASSIFICATION_MAP:
            return LANGUAGE_CLASSIFICATION_MAP[stripped_language]
        LOGGER.debug("Finished message language detection: %s", stripped_language)
        return stripped_language

    def classify_language(self, message):
        """
        Check if a message fits the estimated language.
        Return another language tag, if it does not fit.
        """
        prompt = LlmPrompt(
            settings.LANGUAGE_CLASSIFICATIONH_MODEL,
            [
                LlmMessage(Prompts.LANGUAGE_CLASSIFICATION, role="system"),
                LlmMessage(message, role="user")
            ],
            json_schema = Prompts.LANGUAGE_CLASSIFICATION_SCHEMA
        )
        LOGGER.debug("Detecting message language")
        response = LlmResponse(self.llm_api.chat_prompt(prompt)).as_dict()
        return self.parse_language(response)

    def is_numerical(self, message):
        """
        Check if message is numerical
        """
        return re.match(r"^[0-9\s+\.\,]*$", message)

    def check_cache(
        self, source_language: str, target_language: str, message: str
    ) -> tuple[str, str | None]:
        """
        Check if message exists in translation cache. If not, return cache key
        """
        cache_key = hashlib.sha256(
            f"{source_language}-{target_language}-{message}".encode("utf-8")
        ).hexdigest()
        if translated_message := cache.get(cache_key):
            return cache_key, translated_message
        return cache_key, None

    def translation_required(
        self, source_language: str, target_language: str, message: str
    ) -> bool:
        """
        Check if a translation is (not) required.
        """
        if source_language == target_language:
            LOGGER.debug(
                "Skipping translation from %s to %s", source_language, target_language
            )
            return False
        if self.is_numerical(message):
            return False
        return True

    def chunked_translation_pipeline(
        self, source_language: str, target_language: str, message: str
    ) -> str:
        """
        Translate text in chunks (required for NLLB)
        """
        pipe = pipeline("translation", model=settings.TRANSLATION_MODEL)
        return " ".join(
            [
                result["translation_text"]
                for result in pipe(
                    self.split_text(message),
                    tgt_lang=LANGUAGE_MAP[target_language],
                    src_lang=LANGUAGE_MAP[source_language],
                )
            ]
        )

    def translate_message(
        self, source_language: str, target_language: str, message: str
    ) -> str:
        """
        Translate a message from source to target language
        """
        if not self.translation_required(source_language, target_language, message):
            return message
        cache_key, translated_message = self.check_cache(
            source_language, target_language, message
        )
        if translated_message is not None:
            return translated_message
        try:
            LOGGER.debug(
                "Starting translation from %s to %s", source_language, target_language
            )
            translated_message = self.chunked_translation_pipeline(
                source_language, target_language, message
            )
            LOGGER.debug(
                "Finished translation from %s to %s", source_language, target_language
            )
        except KeyError as exc:
            raise KeyError(
                f"Language pair ({source_language}, {target_language})"
                f" not supported by translation model"
            ) from exc
        cache.set(cache_key, translated_message)
        return translated_message

    def opportunistic_translate(self, expected_language, message):
        """
        Translate if detected language does not fit the expected language
        """
        classified_language = self.classify_language(message)
        return (
            message
            if classified_language == expected_language
            else self.translate_message(classified_language, expected_language, message)
        )

    def split_text(self, text, max_length=200, lang="xx"):
        """
        Chunk text into max_length char chunks while keeping complete sentences.
        Supports multi-lingual splitting using spacy's multi-lingual model,
        see - https://spacy.io/models/xx
        """
        try:
            nlp = spacy.load(lang)
        except:
            nlp = spacy.load("xx_sent_ud_sm")

        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        LOGGER.debug("Translation chunks:\n - %s", "\n - ".join(chunks))
        return chunks
