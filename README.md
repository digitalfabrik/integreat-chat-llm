# About

RAG/LLM supported online migration counseling service & improved Integreat search engine. It integrates as a chat service into the [Integreat App](https://github.com/digitalfabrik/integreat-app) and presents requests in a Zammad to counselors. The solution aims to be privacy friendly by not using any third party LLM services.

# Start Project
1. Install a virtual environment and activate it
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```
1. Install all dependencies
   ```
   pip install .
   ```
1. Run the server:
   ```
   cd integreat_chat
   python3 manage.py migrate
   python3 manage.py runserver
   ```
