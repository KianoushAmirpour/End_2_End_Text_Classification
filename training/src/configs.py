from typing import Dict, Any
from dataclasses import dataclass, field

@dataclass
class BaseConfigs:
    params: Dict[str, Any] = field(default_factory=dict)
        
    def to_dict(self):
        return self.params

@dataclass       
class LogisticRegressionConfig(BaseConfigs):
    params: Dict[str, Any] = field(default_factory=dict)
  
@dataclass  
class RandomForestConfig(BaseConfigs):
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class XGboostConfig(BaseConfigs):
    params: Dict[str, Any] = field(default_factory=dict)

    
class LlamaThreeSettings:
    model_name = 'Meta-Llama-3-8B-Instruct.Q5_K_M.gguf'
    model_configs = {
        'n_ctx': 8192,
        'n_gpu_layers': -1,
        "verbose": False
    }
    text_generation_configs = {'max_tokens': 30, 'temperature': 0.2, "top_p": 0.1,
                               'echo': False, 'seed': 11111, 'stream': False}

    few_shot_text_classification_prompt = """
        You are a helpful AI assistant which classify comments as sincere or insincere.
        You will be provided with some EXAMPLES of comments and their labels.
        You Must answer with just one word as the label:sincere or insincere.
        Never use any other label or add anything else.
        
        the provided EXAMPLES are below:\n
        example 1:
        comment: How did Quebec nationalists see their province as a nation in the 1960s?
        label: sincere\n
        example 2:
        comment: What are the symptoms of skin disease?
        label: sincere\n
        example 3:
        comment: Does Chinese people hate Filipinos?
        label: insincere\n
        example 4:
        comment: Can women ever be talented and brilliant?
        label: insincere\n
        example 5:
        comment: How good is the photo quality of Nokia?
        label: sincere\n
        example 6:
        comment: Do girls find short and skinny guys timid?
        label: insincere\n
        example 7:
        comment:What are some Spanish rice paella recipes?
        label: sincere\n
        example 8:
        comment: Are there any Muslims in India who opted for vasectomy?
        label: insincere\n
        
        Having the above examples, classify the below Question as sincere or insincere\n.
        DO NOT GENERATE ANOTHER QUESTION\n.
        <|eot_id|><|start_header_id|>user<|end_header_id|>\n 
        question: {query}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
""" 
