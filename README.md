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
