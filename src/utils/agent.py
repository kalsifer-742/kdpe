from mistralai.client import Mistral

class Agent:
    def __init__(self, api_key, model, temperature: float, format):
        self.client = Mistral(api_key)
        self.model = model
        self.temperature = temperature
        self.format = format
        self.history = []

    def get_history(self):
        return self.history

    def set_format(self, format):
        self.format = format

    #TODO look into mistral API for conversations
    def chat(self, message):
        self.history.append(message)

        response = self.client.chat.parse(
            model = self.model,
            messages = self.history,
            response_format = self.format,
            temperature = self.temperature
        )

        self.history.append({"role": "assistant", "content": response.choices[0].message.content})
        parsed_response = response.choices[0].message.parsed

        return parsed_response.model_dump()
    
    def chat_batch(self, messages):
        response = self.client.chat.parse(
            model = self.model,
            messages = messages,
            response_format = self.format,
            temperature = self.temperature
        )

        parsed_response = response.choices[0].message.parsed

        return parsed_response.model_dump()