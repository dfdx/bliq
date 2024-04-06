from typing import List

from llama_index.core.schema import TextNode
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from kava.postgres import StoredTextNode, create_engine_from_env

DEFAULT_EMBEDDER_MODEL_ID = "all-MiniLM-L6-v2"


class PGStore:
    """
    PostgreSQL-based vector store
    """

    def __init__(
        self,
        embedder: SentenceTransformer | str = DEFAULT_EMBEDDER_MODEL_ID,
        conn_str=None,
    ):
        if isinstance(embedder, str):
            embedder = SentenceTransformer(embedder)
        self.embedder = embedder or SentenceTransformer(DEFAULT_EMBEDDER_MODEL_ID)
        self.embedding_size = self.embedder.encode("hello").shape[0]
        self.engine = create_engine(conn_str) if conn_str else create_engine_from_env()

    def add(self, nodes: List[TextNode]):
        for node in nodes:
            if not node.embedding:
                node.embedding = self.embedder.encode(node.text).tolist()
        stored_nodes = [StoredTextNode.from_text_node(node) for node in nodes]
        with Session(self.engine) as session:
            session.add_all(stored_nodes)
            session.commit()

    def find(self, question_or_embedding: str | List, limit: int = 10):
        if isinstance(question_or_embedding, str):
            embedding = self.embedder.encode(question_or_embedding)
        else:
            embedding = question_or_embedding
        with Session(self.engine) as session:
            query = select(StoredTextNode)
            query = query.order_by(StoredTextNode.embedding.cosine_distance(embedding))
            query = query.limit(limit)
            s_nodes = session.scalars(query).all()
        return [s_node.to_text_node() for s_node in s_nodes]


def main():
    self = PGStore()
    texts = """Appalachian Spring is an American ballet created by the composer Aaron Copland and the choreographer Martha Graham (pictured), later arranged as an orchestral work. Copland composed the ballet for Graham upon a commission from Elizabeth Sprague Coolidge. Set in a 19th-century settlement in Pennsylvania, the ballet follows the Bride and the Husbandman as they get married and celebrate with the community. The original choreography was by Graham, with costumes by Edythe Gilfond and sets by Isamu Noguchi. The ballet was well-received at the 1944 premiere, earning Copland the Pulitzer Prize for Music during its 1945 United States tour. """.split(
        ". "
    )
    nodes = [TextNode(text=text) for text in texts]
    self.add(nodes)

    question_or_embedding = "When does the ballet take place?"
    limit = 3
    nodes = self.find(question_or_embedding)
    session = Session(self.engine)
