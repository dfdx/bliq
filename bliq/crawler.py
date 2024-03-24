from typing import List
from urllib.parse import urljoin, urlparse
import multiprocessing as mp
import multiprocessing.queues as queues
import logging
import asyncio
import httpx
import bs4


# async def fetch(client: httpx.AsyncClient, outq: mp.Queue, url: str, params: dict = None):
#     params = params or {}

logger = logging.getLogger(__name__)


async def download_job_coro(inq: queues.Queue, outq: queues.Queue, rate_limit=None):
    async with httpx.AsyncClient() as client:
        while True:
            req_dict = inq.get()
            req = client.build_request(**req_dict)
            logger.debug(f"Requesting {req.url}")
            resp = await client.send(req)
            outq.put(resp)
            if rate_limit:
                await asyncio.sleep(1 / rate_limit)


def download_job(inq: queues.Queue, outq: queues.Queue, **kwargs):
    """
    Run job that consumes requests and produces responses.

    Example:
    ```
    inq = mp.Queue()
    outq = mp.Queue()
    kwargs = {}
    p = mp.Process(target=download_job, args=(inq, outq,), kwargs=kwargs)
    p.start()
    for url in start_urls:
        inq.put({"method": "GET", "url": url})
    while outq.qsize() > 0:
        resp = outq.get()
        print(resp.status_code)
    ```

    Parameters
    ----------
    inq : mp.queues.Queue
        Mutliprocessing queue of request dicts
    outq : mp.queues.Queue
        Multiprocessing queue of httpx.Response
    kwargs
        Keyword arguments passed to download_job_coro()
    """
    asyncio.run(download_job_coro(inq, outq, **kwargs))


def crawl(start_urls: List[str], **kwargs):
    inq = mp.Queue()     # queue of dicts with args to AsyncClient.build_request()
    outq = mp.Queue()    # queue of httpx.Response
    p = mp.Process(target=download_job, args=(inq, outq,), kwargs=kwargs)
    p.start()
    # TODO: deduplicate
    # TODO: implement create_download_job()
    for url in start_urls:
        inq.put({"method": "GET", "url": url})
    while inq.qsize() > 0 or outq.qsize() > 0:
        resp = outq.get()
        if resp.status_code != 200:
            continue
        html = resp.text
        soup = bs4.BeautifulSoup(html, "html.parser")
        links = soup.find_all("a")
        for link in links:
            path = link.get("href")
            if path and path.startswith("/"):
                path = urljoin(url, path)
            if resp.url.netloc.decode("utf-8") == urlparse(path).netloc:
                inq.put({"method": "GET", "url": path})
        yield html
    p.terminate()



def main():
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)
    start_urls = ["https://www.nhs.uk/"]
    kwargs = {"rate_limit": 20}
    it = crawl(start_urls, **kwargs)
    for html in it:
        print(f"HTML of length {len(html)}")
