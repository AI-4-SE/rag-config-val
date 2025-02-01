from openai import OpenAI, RateLimitError, Timeout, APIError, APIConnectionError
from rich.logging import RichHandler
from typing import Tuple, List, Dict
from ollama._types import ResponseError, RequestError
from src.util import get_dominat_response
import ollama
import backoff
import logging
import os
import json
import re


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()],
)

class GeneratorFactory:
    gpt_model_names = ["gpt-4o-mini-2024-07-18", "gpt-4o-2024-11-20", "gpt-4-turbo-2024-04-09"]
    ollama_model_names = ["llama3.1:70b ", "deepseek-r1:7b", "deepseek-r1:14b", "deepseek-r1:70b"]

    def get_generator(self, model_name: str, temperature: int):
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
    def __init__(self, model_name: str, temperature: int) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.answer_pool = []
        self.max_genrations = 3

    def generate(self, messages) -> str:
        """Generate responses and return the most dominant response."""
        response_pool = []

        for _ in range(self.max_genrations):
            response = self._generate(messages=messages)
            print("Reponse: ", response)
            parsed_response = self._parse_response(response=response)
            response_pool.append(parsed_response)

        # Get dominant responses#
        print("Response: Pool \n")
        for x in response_pool:
            print(x)

        dominant_response = get_dominat_response(responses=response_pool)

        return {
            self.model_name: dominant_response
        }
            

    @backoff.on_exception(
        backoff.expo,
        (
            RateLimitError,
            APIError,
            APIConnectionError,
            Timeout,
            Exception
        ),
        max_tries=5
    )
    def _generate(self, messages: List) -> str:
        pass

    def _parse_response(self, response: str):
        """Parse generation reponse into a dictionary."""
        try:
            pattern = r'{\n.*?\n}'
            matches = re.findall(pattern, response, re.DOTALL)
            if len(matches) != 1:
                print(f"Error: {response} is not a valid json string.")
                return None
            
            response_json = json.loads(matches[0])

            if self._is_valid_generation(response_json):
                return response_json
            else:
                print(f"Error: {response} does not contain the expected keys.")
                return None
            
        except:
            return None


    def _is_valid_response(self, candidate: Dict) -> bool:
        return candidate.get("plan") and candidate.get("rational") and candidate.get("isDependency")




class GPTGenerator(Generator):
    def __init__(self, model_name: str, temperature: int) -> None:
        super().__init__(
            model_name=model_name, 
            temperature=temperature
        )
        logging.info(f"GPT ({model_name}) generator initialized.")

    def _generate(self, messages: List) -> str:
        client = OpenAI(
            api_key=os.getenv("OPENAI_KEY"),
            base_url=os.getenv()
        )
        response = client.chat.completions.create(
            model=self.model_name, 
            messages=messages,        
            temperature=self.temperature,
            response_format={"type": "json_object"},
            max_tokens=1000
        )
    
        response_content = response.choices[0].message.content

        if not response or len(response_content.strip()) == 0:
            raise Exception("Response content was empty.")
        
        return response_content


class OllamaGenerator(Generator):
    def __init__(self, model_name: str, temperature: int) -> None:
        super().__init__(
            model_name=model_name, 
            temperature=temperature
        )
        logging.info(f"Ollama ({model_name}) generator initialized.")


    def _generate(self, messages: List) -> Tuple:
        response = ollama.chat(
            model=self.model_name, 
            messages=messages,
            format="json",
            options={
                "temperature": self.temperature
            }
        )
        return response['message']['content']
