from hermes.models.openai import OpenAIChat


def get_llm():
    return OpenAIChat(
        model="gpt-4o-mini",
        temperature=0.2
    )