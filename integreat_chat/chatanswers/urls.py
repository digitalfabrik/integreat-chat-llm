from django.urls import path
from . import views

urlpatterns = [
    path("extract_answer/", views.extract_answer, name="extract_answer"),
    path("search_documents/", views.search_documents, name="search_documents"),
    path("translate_message/", views.translate_message, name="translate_message"),
    path("update_vdb/", views.update_vdb, name="update_vdb")
]
