from openai import OpenAI, RateLimitError, Timeout, APIError, APIConnectionError
from rich.logging import RichHandler
from typing import List, Dict
from src.utils import get_dominat_response
import ollama
import backoff
import logging
import os
import json
import re
import traceback

DEFAULT_RESPONSE = {"plan": "None", "rationale": "None", "isDependency": "None"}


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)


class GeneratorFactory:
    gpt_model_names = ["gpt-4o-mini-2024-07-18", "gpt-4o-2024-11-20"]
    ollama_model_names = [
        "llama3.1:70b", 
        "llama3.1:8b", 
        "deepseek-r1:14b", 
        "deepseek-r1:70b", 
        "phi4:latest"
    ]

    def get_generator(self, model_name: str, temperature: float = 0.5):
        if model_name in self.gpt_model_names:
            return GPTGenerator(
                model_name=model_name,
                temperature=temperature
            )
        if model_name in self.ollama_model_names:
            return OllamaGenerator(
                model_name=model_name,
                temperature=temperature
            )
    
        else:
            raise Exception(f"Model {model_name} is not yet supported.")
        

class Generator:
    def __init__(self, model_name: str, temperature: float = 0.5, max_tokens: int = 500) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.answer_pool = []
        self.max_genrations = 3
        self.keys = ["plan", "rationale", "isDependency"]

    def generate(self, messages) -> dict:
        """Generate responses and return the most dominant response."""
        response_pool = []

        for _ in range(self.max_genrations):
            try:   
                 print(f"Generate response using {self.model_name}")
                 response = self._generate(messages=messages)
            except Exception as error:
                 print(f"Exception occurred: {error}. Fall back to DEFAULT RESPONSE")
                 print(traceback.print_exc())
                 response = DEFAULT_RESPONSE.copy()
                 response["error"] = str(error)

            print("Add response to answer pool.")
            response_pool.append(response)

        # Get dominant responses#
        print("Get dominant response.")
        dominant_response = get_dominat_response(responses=response_pool)

        return dominant_response
    
    def generate_with_ratings(self, messages) -> dict:
        """Generate responses and return the most dominant response."""
        response_pool = []
        ratings = []

        for _ in range(self.max_genrations):
            try:   
                 print(f"Generate response using {self.model_name}")
                 response = self._generate(messages=messages)
                 ratings.append(response["isDependency"])
            except Exception as error:
                 print(f"Exception occurred: {error}. Fall back to DEFAULT RESPONSE")
                 print(traceback.print_exc())
                 response = DEFAULT_RESPONSE.copy()
                 response["error"] = str(error)

            print("Add response to answer pool.")
            response_pool.append(response)

        # Get dominant responses#
        print("Get dominant response.")
        dominant_response = get_dominat_response(responses=response_pool)

        return dominant_response, ratings
    
    def _generate(self, messages: List) -> str:
        pass

    def _parse_response(self, response: str):
        """Parse generation reponse into a dictionary."""
        pattern = r'{\n.*?\n}'
        matches = re.findall(pattern, response, re.DOTALL)
        if len(matches) == 1:
            response_json = json.loads(matches[0], strict=False)
        else:
            response_json = json.loads(response, strict=False)

        return response_json
    
    def _are_keys_available(self, response_dict: Dict) -> bool:
        if not all(key in response_dict for key in self.keys):
            return False
        else:
            return True
    

class GPTGenerator(Generator):
    def __init__(self, model_name: str, temperature: float = 0.5, max_tokens: int = 500) -> None:
        super().__init__(
            model_name=model_name, 
            temperature=temperature,
            max_tokens=max_tokens
        )
        print(f"GPT ({model_name}) generator initialized.")

    @backoff.on_exception(
        backoff.expo,
        (
            RateLimitError,
            APIError,
            APIConnectionError,
            Timeout,
            Exception,
            json.JSONDecodeError
        ),
        max_tries=3
    )
    def _generate(self, messages: List) -> str:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            #base_url=os.getenv("BASE_URL")
        )
        response = client.chat.completions.create(
            model=self.model_name, 
            messages=messages,        
            temperature=self.temperature,
            response_format={"type": "json_object"},
            max_tokens=self.max_tokens,
            timeout=90
        )

        response_content = response.choices[0].message.content.strip()
        response_dict = self._parse_response(response=response_content)

        # Check if all keys are present in the repsonse dict
        if not self._are_keys_available(response_dict=response_dict):
            raise Exception("Response content is missing required keys")

        return response_dict


class OllamaGenerator(Generator):
    def __init__(self, model_name: str, temperature: int) -> None:
        super().__init__(
            model_name=model_name, 
            temperature=temperature
        )
        print(f"Ollama ({model_name}) generator initialized.")
        self.client = ollama.Client(timeout=180)

    @backoff.on_exception(
        backoff.expo,
        (
            Exception,
            json.JSONDecodeError
        ),
        max_tries=3
    )
    def _generate(self, messages: List) -> Dict:
        response = self.client.chat(
            model=self.model_name, 
            messages=messages,
            format="json",
            options={
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
        )

        response_content = response['message']['content'].strip()
        response_dict = json.loads(response_content)

        if not self._are_keys_available(response_dict=response_dict):
            raise Exception("Response content is missing required keys")
        
        return response_dict
