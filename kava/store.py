import os
from typing import List
from collections import OrderedDict

import numpy as np
import psycopg
from psycopg.types.json import Jsonb
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from llama_index.core.schema import TextNode
from tqdm import tqdm


DEFAULT_EMBEDDER_MODEL_ID = "all-MiniLM-L6-v2"


def getenv_or_raise(name: str):
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"Attemping to load connection info from environment, " +
            f"but {name} variable is not set")
    else:
        return value

def connection_string_from_env():
    names = ("PGHOST", "PGDBNAME", "PGUSER", "PGPASSWORD")
    host, dbname, user, password = [
        getenv_or_raise(name) for name in names
    ]
    return f"host={host} dbname={dbname} user={user} password={password}"


class PGStore:
    """
    PostgreSQL-based vector store
    """

    # map from TextNode fields to table columns
    attr2col = OrderedDict({
        "id_": "id",
        "text": "text",
        "embedding": "embedding",
        "metadata": "metadata",
        "start_char_idx": "start_char_idx",
        "end_char_idx": "end_char_idx"
    })

    def __init__(self, table: str, embedder: SentenceTransformer | str = DEFAULT_EMBEDDER_MODEL_ID, conn_info=None):
        if isinstance(embedder, str):
            embedder = SentenceTransformer(embedder)
        self.embedder = embedder or SentenceTransformer(DEFAULT_EMBEDDER_MODEL_ID)
        self.embedding_size = self.embedder.encode("hello").shape[0]
        self.table = table
        self.conn_info = conn_info or connection_string_from_env()
        self.reconnect()

    def reconnect(self):
        self.conn = psycopg.connect(self.conn_info, row_factory=dict_row)
        register_vector(self.conn)

    def ensure_initialized(self):
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                id UUID PRIMARY KEY,
                text TEXT,
                embedding VECTOR({self.embedding_size}),
                metadata JSONB,
                start_char_idx INT,
                end_char_idx INT
            )"""
        )
        self.conn.commit()


    def add(self, nodes: List[TextNode]):
        self.ensure_initialized()
        for node in nodes:
            if not node.embedding:
                node.embedding = self.embedder.encode(node.text).tolist()
        sql = f"""INSERT INTO {self.table}
            (id, text, embedding, metadata, start_char_idx, end_char_idx)
        VALUES
            (%s, %s, %s, %s, %s, %s)
        """
        for node in nodes:
            self.conn.execute(sql, (node.id_, node.text, node.embedding, Jsonb(node.metadata), node.start_char_idx, node.end_char_idx))
        self.conn.commit()


    def _find(self, embedding: np.array, limit: int = 10):
        self.ensure_initialized()
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, text, embedding, metadata, start_char_idx, end_char_idx FROM {self.table}
                ORDER BY embedding <-> %s LIMIT {limit}
                """,
                (embedding,),
            )
            records = cur.fetchall()
        nodes = []
        for rec in records:
            node = TextNode(
                id_=str(rec["id"]),
                text=rec["text"],
                embedding=rec["embedding"].tolist(),
                metadata=rec["metadata"],
                start_char_idx=rec["start_char_idx"],
                end_char_idx=rec["end_char_idx"]
            )
            nodes.append(node)
        return nodes

    def find(self, question: str, limit: int = 10):
        embedding = self.embedder.encode(question)
        return self._find(embedding, limit=limit)



def main():
    TEXT = """The 1876 association football match between the national teams representing Scotland and Wales was the first game played by the latter side. It took place on 25 March 1876 at Hamilton Crescent, Partick, the home ground of the West of Scotland Cricket Club. The match was also the first time that Scotland had played against a side other than England."""
    embedder = DEFAULT_EMBEDDER_MODEL_ID
    table = "medical_embeddings"
    self = PGStore(table, embedder=embedder)
    for node in self.find("When did it happen?"):
        print(node.text)

    nodes = [TextNode(text=piece) for piece in TEXT.split(".")]

