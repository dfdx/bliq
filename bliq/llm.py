from typing import Iterable

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    pipeline,
)


class LLM:
    """
    Simple wrapper for Hugginface generative text models.

    Usage:

    ```
    llm = LLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    out = llm.generate("Once upon a time")
    print(out[0])
    ```
    """

    default_gen_kwargs = {
        "max_new_tokens": 512,
        "return_full_text": False,
    }

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    @staticmethod
    def from_pretrained(model_id: str, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        return LLM(model=model, tokenizer=tokenizer)

    def generate(self, texts: str | Iterable[str], **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        kwargs = self.default_gen_kwargs | kwargs
        out = self.pipe(texts, **kwargs)
        out = (x[0]["generated_text"] for x in out)
        # for str and list, return list; for other iterators return iterator
        out = list(out) if isinstance(texts, list) else out
        return out
