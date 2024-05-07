import gradio as gr
import os
import requests
import json


def respond(message, history):
    if len(message.strip()) == 0:
        return "ERROR the question should not be empty"

    url = os.getenv('API_END_POINT')
    bearer_token = os.getenv('API_TOKEN')

    if bearer_token is None or url is None:
        return "ERROR missing authorization configurations"

    headers = {
        'Authorization': f'Bearer {bearer_token}', 
        'Content-Type': 'application/json'
    }

    query = {
          "dataframe_split": {
            "columns": [
              "query"
            ],
            "data": [
              [message]
            ]
          }
        }
    
    data_json = json.dumps(query, allow_nan=True)

    try:
        response = requests.request(method='POST', headers=headers, url=url, data=data_json)
        response_data = response.json()
        response_data=response_data["predictions"][0]

    except Exception as error:
        response_data = f"ERROR status_code: {type(error).__name__}"

    return response_data

    
"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
movie_recommender = gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(show_label=False, container=False, show_copy_button=True, bubble_full_width=True),
    textbox=gr.Textbox(placeholder="Ask me a movie recommendation",
                       container=False, scale=7),
    title="Movie Recommendation RAG powered by Llama-3",
    description="""This LLM chatbot is intended to generate movie recommendations for OTT platforms from the list of labels available with the platform. 
    It is using Llama-3 as a generation model & databricks-bge-large-en for embedding.
    This application is built as part of learning Gen AI with databricks for a hackathon & can hallucinate. Thus it should not be used as production content.
    """,
    examples=[["Suggest me the best horror movie, that can scare anyone."],
              ["I liked 'Jurassic Park'. Can you suggest some movies similar?"],
              ["I am feeling bored, suggest me some movies to improve my mood."],
              ["My friends are coming over, suggest me some movies to chill with them."],],
    cache_examples=False,
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear",
)


if __name__ == "__main__":
    movie_recommender.launch()