"""
Retrieving matching documents for question an create summary text
"""
import logging

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

from django.conf import settings

from .search import SearchService
from .language import LanguageService
from ..static.prompts import Prompts
from ..static.messages import Messages

LOGGER = logging.getLogger('django')


class AnswerService:
    """
    Service for providing summary answers to question-like messages.
    """
    def __init__(self, region, language):
        self.language = language
        self.region = region
        self.llm_model_name = settings.RAG_MODEL
        self.llm = self.load_llm(self.llm_model_name)

    def load_llm(self, llm_model_name):
        """
        Prepare LLM
        """
        return Ollama(model=llm_model_name, base_url=settings.OLLAMA_BASE_PATH)

    def optimize_query_for_retrieval(self, message):
        """
        Optimize the user query for document retrieval
        """
        prompt = PromptTemplate.from_template(Prompts.OPTIMIZE_MESSAGE)
        chain = prompt | self.load_llm(settings.RAG_QUERY_OPTIMIZATION_MODEL) | StrOutputParser()
        return chain.invoke({"message": message})

    def needs_answer(self, message):
        """
        Check if a chat message is a question
        """
        prompt = PromptTemplate.from_template(Prompts.CHECK_QUESTION)
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"message": message})
        if answer.startswith("Yes"):
            return True
        return False

    def extract_answer(self, question):
        """
        Create summary answer for question
        """
        search = SearchService(self.region, self.language)
        results = search.search_documents(
            question,
            include_text=True
        )
        results = search.retrieve_unique_pages(results, settings.RAG_MAX_PAGES)

        LOGGER.debug("Number of retrieved documents: %i", len(results))
        if settings.RAG_RELEVANCE_CHECK:
            results = [result for result in results if self.check_document_relevance(
                question, result['text']
            )]
        LOGGER.debug("Number of documents after relevance check: %i", len(results))

        context = RunnableLambda(lambda _: "\n".join(
            [result['text'] for result in results]
        )[:settings.RAG_CONTEXT_MAX_LENGTH])
        if not results:
            language_service = LanguageService()
            return {
                "answer": language_service.translate_message(
                    "en", self.language,
                    Messages.NO_ANSWER
                )
            }
        rag_chain = (
            {"context": context, "question": RunnablePassthrough()}
                | settings.RAG_PROMPT
                | self.llm
                | StrOutputParser()
        )
        answer = rag_chain.invoke(question)
        return {
            "answer": answer,
            "sources": [result['source'] for result in results],
            "details": [{
                "context": result['text'],
                "score": result['score']
            } for result in results],
        }


    def check_document_relevance(self, question, content):
        """
        Check if the retrieved documents are relevant
        """
        grade_prompt = PromptTemplate.from_template(Prompts.RELEVANCE_CHECK)
        chain = grade_prompt | self.llm | StrOutputParser()

        response = chain.invoke({"document": content, "question": question})
        response = response.strip().lower()
        return response.startswith("yes")
