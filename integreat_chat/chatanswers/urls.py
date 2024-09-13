from django.urls import path
from . import views

urlpatterns = [
    path("extract_answer/", views.extract_answer, name="extract_answer"),
    path("update_vdb/", views.update_vdb, name="update_vdb")
]
