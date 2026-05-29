import os
from pathlib import Path
from mistralai.client import Mistral


API_KEY = [...]

client = Mistral(api_key=api_key)

class Agent:
    def __init__(self, model):
        self.history = []
        self.model = model

    def get_history(self):
        return self.history

    def chat(self, message):
        self.history.append(message)

        chat_response = client.chat.complete(
            model = self.model,
            messages = [
                {
                    "role": "user",
                    "content": "What is the best French cheese?",
                },
            ],
            response_format=schema.model_json_schema()
            temperature=0.2
        )

        response = ollama.chat(
            model = "gemma4:e4b", #26b is the limit on my machine
            messages = self.history,
            format = Schema.model_json_schema(),
            options = {"temperature": 0.2}
        )
        self.history.append({"role": "assistant", "content": response['message']['content']})

        schema = json.loads(response['message']['content'])
        print_schema(schema['entities'], schema['relationships'])
        
        console.print(f"[bold cyan]Agent[/bold cyan] >>> {schema['clarifying_question']}")