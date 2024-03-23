from typing import Generator

import torch
from bliq.llm import LLM


MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"


def test_llm():
    llm = LLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    out1 = llm.generate("Once upon a time")
    assert isinstance(out1, list)
    out2 = llm.generate(["Once upon a time"])
    assert isinstance(out2, list)
    out3 = llm.generate(t for t in ["Once upon a time"])
    assert isinstance(out3, Generator)
