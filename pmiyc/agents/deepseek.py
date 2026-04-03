import copy
import openai
import os
import random
from pmiyc.agents.agents import Agent
import time
from copy import deepcopy

class DeepSeekAgent(Agent):
    def __init__(
        self,
        agent_name: str,
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=400,
        seed=None,
        base_url="https://api.deepseek.com/v1",
        **kwargs
    ):
        super().__init__(agent_name)
        self.run_epoch_time_ms = str(round(time.time() * 1000))
        self.model = model
        self.conversation = []
        self.prompt_entity_initializer = "system"
        self.seed = (
            int(self.run_epoch_time_ms) + random.randint(0, 2**16)
            if seed is None
            else seed
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
        )

    def __deepcopy__(self, memo):
        """
        Deepcopy is needed because we cannot pickle the client object.
        :param memo:
        :return:
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if (type(v)) == type(self.client):
                v = "ClientObject"
            setattr(result, k, deepcopy(v, memo))
        return result

    def chat(self):
        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        
        return chat_completion.choices[0].message.content
        
    def update_conversation_tracking(self, role, message):
        self.conversation.append({"role": role, "content": message})
