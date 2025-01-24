from django.urls import path
from . import views

urlpatterns = [
    path("message/", views.translate_message, name="translate_message"),
    path("detect/", views.detect_language, name="detect_language"),
]
