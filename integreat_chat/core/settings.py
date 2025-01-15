"""
Django settings for integreat_chat project.

Generated by 'django-admin startproject' using Django 5.0.6.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/5.0/ref/settings/
"""

import configparser
import os

from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

config = configparser.ConfigParser()
if os.path.isfile('/etc/integreat-chat.ini'):
    config.read('/etc/integreat-chat.ini')
else:
    config.read(os.path.join(__file__, "../../integreat-chat.ini"))

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-u*mc*mu#()r6qdrotm(1=+kuo!2-*76fav*-m*m3e8v+hutfn('

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

# Django logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'apache': {
            'level': 'DEBUG' if DEBUG else config["DEFAULT"]["LOG_LEVEL"],
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stderr',  # Logs to Apache's error log
        },
    },
    'loggers': {
        'django': {
            'handlers': ['apache'],
            'level': 'DEBUG' if DEBUG else config["DEFAULT"]["LOG_LEVEL"],
            'propagate': True,
        },
    },
}

ALLOWED_HOSTS = ["127.0.0.1", "igchat-inference.tuerantuer.org"]

INTEGREAT_CMS_DOMAIN = config["DEFAULT"]["INTEGREAT_CMS_DOMAIN"]

# Configuration Variables for answer service
QUESTION_CLASSIFICATION_MODEL = "llama3.3"

LANGUAGE_CLASSIFICATIONH_MODEL = "llama3.3"

TRANSLATION_MODEL = "facebook/nllb-200-3.3B"

RAG_DISTANCE_THRESHOLD = 20
RAG_MAX_PAGES = 3
RAG_MODEL = "llama3.3"
RAG_RELEVANCE_CHECK = True
RAG_QUERY_OPTIMIZATION = True
RAG_QUERY_OPTIMIZATION_MODEL = "llama3.3"
RAG_CONTEXT_MAX_LENGTH = 8000
RAG_SUPPORTED_LANGUAGES = ["en", "de"]
RAG_FALLBACK_LANGUAGE = "en"

# SEARCH_MAX_DOCUMENTS - number of documents retrieved from the VDB
SEARCH_MAX_DOCUMENTS = 15
SEARCH_DISTANCE_THRESHOLD = 30
SEARCH_MAX_PAGES = 10
SEARCH_EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEARCH_EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name=SEARCH_EMBEDDING_MODEL_NAME, show_progress=False
)
SEARCH_EMBEDDING_MODEL_SUPPORTED_LANGUAGES = [
    "ar", "en", "de", "fa", "it", "uk", "fr", "bg", "hr", "ro", "tr", "uk", "vi"
]
SEARCH_FALLBACK_LANGUAGE = "en"

VDB_HOST = "127.0.0.1"
VDB_PORT = "19530"

LITELLM_SERVER=config['LiteLLM']['SERVER']
LITELLM_API_KEY=config['LiteLLM']['API_KEY']

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'integreat_chat.keywords',
    'integreat_chat.translate',
    'integreat_chat.search',
    'integreat_chat.chatanswers',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'integreat_chat.core.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'core.wsgi.application'


# Database
# https://docs.djangoproject.com/en/5.0/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# Password validation
# https://docs.djangoproject.com/en/5.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.0/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.0/howto/static-files/

STATIC_URL = 'static/'

# Default primary key field type
# https://docs.djangoproject.com/en/5.0/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

if not DEBUG:
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.redis.RedisCache",
            "LOCATION": "redis://127.0.0.1:6379",
            "TIMEOUT": 3600 * 24 * 7,
        }
    }
