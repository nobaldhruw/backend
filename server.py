# server.py
import json
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import openai
import os
import uuid
import requests
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any
from queue import Queue, Empty
from threading import Thread
from cachetools import TTLCache

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
CORS(app)

# Set your OpenAI API key here
openai.api_key = os.getenv("OPENAI_API_KEY")


# Defined a QueueCallback, which takes as a Queue object during initialization. Each new token is pushed to the queue.
class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: Any) -> None:
        return self.q.empty()


# Create an in-memory cache to store conversation objects
conversation_cache = TTLCache(maxsize=100, ttl=3600)  # Adjust ttl as needed


@app.route("/api/langchain/stream", methods=["POST"])
def generate_langchain_stream():
    def stream(data):
        question = data["prompt"]

        # Check if the conversation exists in the cache
        conversation = conversation_cache.get("conversation")

        if conversation is None:
            # If not, create a new conversation and store it in the cache
            q = Queue()

            model = ChatOpenAI(
                streaming=True, callbacks=[QueueCallback(q)], temperature=0
            )
            system_text = "You are a helpful assistant"
            system_message_prompt = SystemMessagePromptTemplate.from_template(
                system_text
            )
            human_text = "{human_input}"
            human_message_prompt = HumanMessagePromptTemplate.from_template(human_text)
            prompt = ChatPromptTemplate(
                messages=[
                    system_message_prompt,
                    MessagesPlaceholder(variable_name="chat_history"),
                    human_message_prompt,
                ]
            )
            memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )
            conversation = LLMChain(
                llm=model, prompt=prompt, verbose=True, memory=memory
            )
            # print("After conversation creation: ", conversation)
            # Store the conversation and the queue in the cache
            conversation_cache["conversation"] = conversation
            conversation_cache["queue"] = q

        # Retrieve the queue from the cache
        q = conversation_cache["queue"]

        job_done = object()

        # Create a funciton to call - this will run in a thread
        def task():
            resp = conversation({"human_input": question})
            q.put(job_done)

        # Create a thread and start the function
        t = Thread(target=task)
        t.start()

        content = ""

        # Get each new token from the queue and yield for our generator
        while True:
            try:
                next_token = q.get(True, timeout=1)
                if next_token is job_done:
                    break
                content += next_token
                yield bytes(next_token, "utf-8")
            except Empty:
                continue

    try:
        data = request.get_json()
        return Response(stream(data), content_type="text/plain")

    except Exception as e:
        print("Error generating response:", e)
        return jsonify(error="Error generating response"), 500


@app.route("/api/execute_code", methods=["POST"])
def execute_code():
    data = request.get_json()
    print("I am in server.py, printing received code: \n", data["code_string"])
    message = {
        "channel": "shell",
        "content": {"silent": False, "code": data["code_string"]},
        "header": {"msg_id": str(uuid.uuid1()), "msg_type": "execute_request"},
        "metadata": {},
        "parent_header": {},
    }
    kernel_id = "94e544bd-10d8-4f3f-8e7f-fb3ced036357"
    session_id = "15632802-33b8-4c26-94eb-97b27f1b0c9f"
    jupyter_server_url = f"http://127.0.0.1:5001/execute/{kernel_id}/{session_id}"
    response = requests.post(jupyter_server_url, data=json.dumps(message))
    if response.status_code == 200:
        # Handle the response data (e.g., print or process it)
        return jsonify(response.json())
    else:
        # Handle the case when the request is not successful
        print(f"Request failed with status code {response.status_code}")
        return jsonify(error="Error generating response"), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.environ.get("PORT", 5000), debug=True)
