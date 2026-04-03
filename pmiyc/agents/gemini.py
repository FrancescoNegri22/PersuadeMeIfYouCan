import os
from google import genai
from pmiyc.agents.agents import Agent
import time
from copy import deepcopy

class GeminiAgent(Agent):
    def __init__(
        self,
        agent_name: str,
        model="gemini-2.5-flash",
        temperature=0.7,
        max_tokens=400,
        **kwargs
    ):
        super().__init__(agent_name)
        self.run_epoch_time_ms = str(round(time.time() * 1000))
        self.model_name = model
        self.conversation = []
        self.prompt_entity_initializer = "system"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    def __deepcopy__(self, memo):
        """
        Deepcopy is needed because we cannot pickle the model object.
        :param memo:
        :return:
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "client" and not isinstance(v, str):
                v = "ClientObject"
            setattr(result, k, deepcopy(v, memo))
        return result

    def chat(self):
        # Convert conversation to Gemini contents format
        contents = []
        for msg in self.conversation:
            role = 'user' if msg['role'] == 'user' else 'model'
            contents.append({
                "role": role,
                "parts": [{"text": msg['content']}]
            })
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents
        )

        return response.text

    def update_conversation_tracking(self, role, message):
        self.conversation.append({"role": role, "content": message})
