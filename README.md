# About

RAG/LLM supported online migration counseling service & improved Integreat search engine. It integrates as a chat service into the [Integreat App](https://github.com/digitalfabrik/integreat-app) and presents requests in a Zammad to counselors. The solution aims to be privacy friendly by not using any third party LLM services.

The Django apps developed in this repo could be moved into the [Integreat CMS](https://github.com/digitalfabrik/integreat-cms) in the future. For the time being the code is separated for faster iteration and testing.

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

# Configuration

## Back End

* Deploy as normal Django application. No database is needed.

## Zammad Integration

To integrate Zammad, the following configuration has to be set:

* Webhook to https://integreat-cms.example.com/api/v3/webhook/zammad/?token=$REGION_TOKEN
* Trigger for webhook:
  * Conditions: `Action is updated`, `Subject contains not "automatically generated message"`
  * Execute: webhook configured above
* Auto response for new tickets in each language, exmple is `EN`
  * Conditions: `State is new`, `Action is updated`, `Subject contains not "automatically generated message"`, `Title contains [EN]`
  * Execute: `Email`, `visibility public`, `Recipient Customer`, `Subject "automatically generated message"`, add a message fitting the needs
* Scheduler to delete old tickets:
  * Run once a week
  * Conditions: `state is closed`, `Last contact before (relative) 6 months`
  * Action: delete
  * Disable Notifications: yes
* Ticket attributes (Admin -> Objects -> Ticket):
  * Name: `automatic_answers`
  * Display: `Automatic Chatbot Answers`
  * Format: `Boolean Field`
  * True as default value
