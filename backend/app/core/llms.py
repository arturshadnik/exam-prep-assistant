from openai import OpenAI
from openai import types
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) 

client = OpenAI(max_retries=0)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def chat_response(messages:list, amount_of_responses:int=1) -> types.completion.Completion:
    response = client.chat.completions.create(
        temperature=0,
        model="gpt.3.5-turbo",
        messages=messages,
        n=amount_of_responses,
    )
    return response

def get_embeddings_batch(list_of_text: list) -> list:
    embeddings = []

    # replace newlines, which can negatively affect performance.
    list_of_text = [text.replace("\n", " ") for text in list_of_text]
    # Split the list_of_text into chunks of size up to 2048
    for i in range(0, len(list_of_text), 2048):
        batch = list_of_text[i:i+2048]
        data = client.embeddings.create(input=batch, model='text-embedding-ada-002').data
        embeddings.extend([d.embedding for d in data])
    return embeddings