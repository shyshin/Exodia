import gradio as gr
from huggingface_hub import InferenceClient
import os

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer






"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

pinecone_client = Pinecone(api_key = os.getenv('PINECONE_API_KEY'))

index = pinecone_client.Index("movies")

def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    # encode user query
    encoded_query = embedding_model.encode(message)

    # retrieve most relevant movie from vector db
    matches = index.query(
        vector= encoded_query.tolist(),
        top_k=1,
        include_metadata = True
    )

    # movie which is most similar
    retrieved_data  = matches['matches'][0]['metadata']['title']

    # Add as context to LLM
    messages.append({"role":"user", "content": retrieved_data})


    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a movie recommender named Exodia. You are extremely reliable. You always mention your name in the beginning of conversation. You will provide me with answers from the given info. Give not more than 5 choices and make sure that answers are complete sentences.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)


if __name__ == "__main__":
    demo.launch()