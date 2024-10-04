"""
A service to detect languages and translate messages
"""

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

#from langchain.runnables.base import RunnableLambda
from langchain_core.runnables import RunnableLambda

from django.conf import settings

class LanguageService:
    """
    Service class that enables searching for Integreat content
    """
    def __init__(self):
        """
        
        """

    def classify_language(self, estimated_lang, message):
        """
        Check if a message fits the estimated language. Return another language tag, if it does not fit.
        """
        prompt = (
            f'Can the following message be in the language with the language tag "{estimated_lang}"? '
            f'If yes, answer with "{estimated_lang}". If no, return a single most likely BCP47 language '
            f'tag for the message.\n\nMessage: {message}'
        )
        parser = StrOutputParser()
        llm = Ollama(model=settings.LANGUAGE_CLASSIFICATIONH_MODEL, base_url=settings.OLLAMA_BASE_PATH)
        answer = parser.invoke(llm.invoke(prompt))
        if answer.startswith(estimated_lang):
            return estimated_lang
        return answer.split("-")[0]

    def translate_message(self, source_language, target_language, message):
        """
        Translate a message from source to target language
        """
        if source_language == target_language:
            return message
        prompt = (
            f'The following message is written in the language with language tag "{source_language}". '
            f'Translate the message to the language with the language tag {target_language}. '
            f'Return only the translated message with no additional words.\n\nMessage: {message}'
        )
        parser = StrOutputParser()
        llm = Ollama(model=settings.TRANSLATION_MODEL, base_url=settings.OLLAMA_BASE_PATH)
        return parser.invoke(llm.invoke(prompt))
