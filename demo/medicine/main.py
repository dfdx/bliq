import logging
import re

from bs4 import BeautifulSoup
from llama_index.core.node_parser.text import SentenceSplitter
from llama_index.core.schema import TextNode
from tqdm import tqdm

from kava.crawler import crawl, get_crawling_process
from kava.llm import LLM
from kava.store import PGStore


def crawl_and_index(
    store: PGStore,
    url: str,
    chunk_size: int = 256,
    chunk_overlap: int = 32,
    **crawl_kwargs,
):
    it = crawl([url], **crawl_kwargs)
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for i, page in enumerate(it):
        text = BeautifulSoup(page.html, "html.parser").get_text()
        text = re.sub(r"\n{3}", "\n\n", text)  # keep at most 2 \n in a row
        chunks = splitter.split_text(text)
        extra_info = {"source": page.url}
        nodes = [TextNode(text=chunk, extra_info=extra_info) for chunk in chunks]
        store.add(nodes)
        if i % 100 == 0:
            print(f"------- Indexed {i} pages -------")


def main():
    logging.basicConfig()
    logging.getLogger("kava.crawler").setLevel(logging.DEBUG)
    url = "https://www.nhs.uk/"
    store = PGStore("medicine_store")
    crawl_and_index(store, url, rate_limit=50)
