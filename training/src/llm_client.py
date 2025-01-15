from pathlib import Path
from llama_cpp import Llama
from .constants import LLM_MODELS_DIR
from typing import Dict
from .logger import setup_logger

logger = setup_logger(__name__)

class LlamaClient:
    def __init__(self, model_settings,
                 query: str,
                 model_path: Path = LLM_MODELS_DIR,
                 )-> None:
      self.model_settings = model_settings
      self.model = Llama(model_path=str(model_path) , **self.model_settings.model_configs)
      self.answer = self._generate_answer_with_few_shot(query=query)
      
          
    def _generate_answer(self, prompt: str):
        answer = self.model(prompt,
                          **self.model_settings.text_generation_configs)
        logger.info(f'Answer generated successfully')
        return answer
        
    def _generate_answer_with_few_shot(self, query: str) -> str:
        prompt = self._generte_prompt(template=self.model_settings.few_shot_text_classification_prompt,
                                          query=query)
        generated_answer = self._generate_answer(prompt)
        cleaned_answer = self.clean_text(generated_answer)
        return cleaned_answer
     
    def _generte_prompt(self, template: str, query: str) -> str:
        prompt = template.format(query=query)
        logger.info(f'Prompt generated: {prompt}')
        return prompt
    
    @staticmethod
    def clean_text(response: Dict) -> str:
        cleaned_text = response['choices'][0]['text']
        logger.info(f'Text cleaned successfully')
        return cleaned_text