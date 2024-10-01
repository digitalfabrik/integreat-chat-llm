"""
Django views
"""

import json
import logging

from integreat_chat.chatanswers.services.answer_service import AnswerService

from .services.milvus import UpdateMilvus
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

LOGGER = logging.getLogger('django')

@csrf_exempt
def extract_answer(request):
    """
    Extract an answer for a user query from Integreat content. Expects a JSON body with message
    and language attributes
    """
    result = None
    if request.method in ('POST') and request.META.get('CONTENT_TYPE').lower() == 'application/json':
        data = json.loads(request.body)
        question = data["message"]
        language = data["language"]
        region = data["region"]
        answer_service = AnswerService(region, language)
        result = { "answer": "" }
        if answer_service.needs_answer(question):
            result = answer_service.extract_answer(question)
    return JsonResponse(result)

@csrf_exempt
def update_vdb(request):
    """
    Extract an answer for a user query from Integreat content. Expects a JSON body with message
    and language attributes
    """
    if request.method in ('POST') and request.META.get('CONTENT_TYPE').lower() == 'application/json':
        data = json.loads(request.body)
        region = data["region"]
        language = data["language"]
        update_milvus = UpdateMilvus(region, language)
        pages = update_milvus.fetch_pages_from_cms()
        text_chunks = []
        for page in pages:
            text_chunks = text_chunks + update_milvus.split_page(page)
        update_milvus.create_embeddings(text_chunks)
    return JsonResponse({"status": "collection updated"})
