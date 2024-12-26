"""
Static prompts
"""

class Prompts:
    """
    Static prompts
    """

    SYSTEM_PROMPT = "You are an internal assistant in an application without user interaction."

    LANGUAGE_CLASSIFICATION = """
Identify the language of the provided message.
Only return the most likely BCP47 language tag that represents the message's language.
Do not add any additional words.

Message: {0}
"""