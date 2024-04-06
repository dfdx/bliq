import asyncio
import logging
import multiprocessing as mp
import multiprocessing.queues as queues
from dataclasses import dataclass
from typing import List
from urllib.parse import urljoin, urlparse

import bs4
import httpx

logger = logging.getLogger(__name__)


async def download_job_coro(
    inq: queues.Queue, outq: queues.Queue, rate_limit=None, retries=3
):
    transport = httpx.AsyncHTTPTransport(retries=retries)
    async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
        while True:
            req_dict = inq.get()
            req = client.build_request(**req_dict)
            logger.debug(f"Requesting {req.url}")
            try:
                resp = await client.send(req)
                outq.put(resp)
            except Exception as e:
                # return pseudo response
                resp = httpx.Response(status_code=500, text=str(e))
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


def create_download_job(**kwargs):
    # queue of dicts with args to AsyncClient.build_request()
    inq: queues.Queue = mp.Queue()
    # queue of httpx.Response
    outq: queues.Queue = mp.Queue()
    p = mp.Process(
        target=download_job,
        args=(
            inq,
            outq,
        ),
        kwargs=kwargs,
    )
    return p, inq, outq


CRAWLING_PROCESSES = []


@dataclass
class WebPage:
    url: str
    html: str


def crawl(start_urls: List[str], **kwargs):
    p, inq, outq = create_download_job(**kwargs)
    global CRAWLING_PROCESSES
    CRAWLING_PROCESSES.append((p, inq, outq))
    p.start()
    submitted_urls: set[str] = set([])
    for url in start_urls:
        inq.put({"method": "GET", "url": url})
    while inq.qsize() > 0 or outq.qsize() > 0:
        resp = outq.get()
        if resp.status_code != 200:
            logger.warning(
                f"Request to {str(resp.url)} failed with code {resp.status_code}"
                + f" and text {resp.text}"
            )
            continue
        page = WebPage(url=str(resp.url), html=resp.text)
        soup = bs4.BeautifulSoup(resp.text, "html.parser")
        links = soup.find_all("a")
        for link in links:
            path = link.get("href")
            if path and path.startswith("/"):
                path = urljoin(url, path)
            # check conditions of skipping
            if resp.url.netloc.decode("utf-8") != urlparse(path).netloc:
                continue
            if path in submitted_urls:
                continue
            # if there's no reason to skip, put new url to the queue
            inq.put({"method": "GET", "url": path})
            submitted_urls.add(path)
        yield page
    if p.is_alive():
        p.terminate()
    CRAWLING_PROCESSES.pop()


def get_crawling_process():
    return CRAWLING_PROCESSES[-1]
