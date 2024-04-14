import logging
import re
import traceback
from typing import List

import torch
from bs4 import BeautifulSoup
from llama_index.core.node_parser.text import SentenceSplitter
from llama_index.core.schema import TextNode
from sqlalchemy import select
from sqlalchemy.engine.base import Engine as SQLEngine
from sqlalchemy.orm import Session
from tqdm import tqdm

from kava.crawler import WebPage, crawl, get_crawling_process
from kava.llm import LLM
from kava.postgres import WebPage as StoredWebPage
from kava.postgres import create_engine_from_env
from kava.store import PGStore

INSERT_BATCH_SIZE = 100


def save_pages(sql_engine: SQLEngine, pages: List[WebPage]):
    s_pages = [StoredWebPage(url=page.url, html=page.html) for page in pages]
    try:
        with Session(sql_engine) as session:
            session.add_all(s_pages)
            session.commit()
    except Exception:
        traceback.format_exc()


def crawl_and_store(sql_engine: SQLEngine, url: str, **crawl_kwargs):
    print(f"Crawling {url}")
    it = crawl([url], **crawl_kwargs)
    pages = []
    count = 0
    for page in it:
        pages.append(page)
        count += 1
        if count % INSERT_BATCH_SIZE == 0:
            save_pages(sql_engine, pages)
            pages = []
            _, inq, outq = get_crawling_process()
            print(
                f"Downloaded {count} pages; inq size = {inq.qsize()}, outq size = {outq.qsize()}"
            )
    # save the rest of pages
    save_pages(sql_engine, pages)
    count += len(pages)
    print(f"Downloaded {count} pages")


def split_and_index(
    sql_engine: SQLEngine,
    url_pattern: str | None = None,
    chunk_size=512,
    chunk_overlap=128,
):
    store = PGStore()
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    with Session(sql_engine) as session:
        stmt = select(StoredWebPage)
        if url_pattern:
            stmt = stmt.where(StoredWebPage.url.like(url_pattern))
        s_pages = [row[0] for row in session.execute(stmt).all()]
        for i, page in enumerate(s_pages):
            text = BeautifulSoup(page.html, "html.parser").get_text(separator=" ")
            text = re.sub(r"\s\s+", " ", text)
            text = re.sub(r"\n\n+", "\n\n", text)  # keep at most 2 \n in a row
            chunks = splitter.split_text(text)
            extra_info = {"source": page.url}
            nodes = [TextNode(text=chunk, extra_info=extra_info) for chunk in chunks]
            store.add(nodes)
            if i % 100 == 0:
                print(f"Indexed {i} pages")


def prepare_data():
    logging.basicConfig()
    # logging.getLogger("kava.crawler").setLevel(logging.DEBUG)
    url = "https://www.nhs.uk/"
    url_pattern = "%www.nhs.uk%"
    sql_engine = create_engine_from_env()
    crawl_kwargs = {"rate_limit": 100}
    crawl_and_store(sql_engine, url, **crawl_kwargs)
    split_and_index(sql_engine, url_pattern=url_pattern)


class GSEngine:

    template = """<s>[INST]
    Given the context, answer the question.

    ### Context:
    {context}

    ### Question:
    {question}
    [/INST]
    """

    def __init__(self, store: PGStore, llm: LLM):
        self.store = store
        self.llm = llm

    def search(self, query: str):
        nodes = self.store.find(query)
        context = "\n-------\n".join(node.text for node in nodes)
        answer = self.llm.generate(self.template.format(context=context, question=query), max_new_tokens=512)[0]
        references = [f"[{i}]: {node.metadata['source']}" for i, node in enumerate(nodes)]
        ref_str = '\n'.join(references)
        return f"""{answer}\n\nReferences: \n{ref_str}"""



def main():
    store = PGStore()
    llm = LLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    self = GSEngine(store, llm)
    query = "I feel pain and see light deformation in my wrist, what can it be?"
    self.search(query)