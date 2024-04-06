import os
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
from llama_index.core.schema import TextNode
from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Index, Integer, String, Text, create_engine
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_SIZE = 384


###############################################################################
#                               Utils & Init                                  #
###############################################################################


def getenv_or_raise(name: str):
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"Attemping to load connection info from environment, "
            + f"but {name} variable is not set"
        )
    else:
        return value


def create_engine_from_env(echo=False):
    names = ("PGHOST", "PGDBNAME", "PGUSER", "PGPASSWORD")
    host, dbname, user, password = [getenv_or_raise(name) for name in names]
    return create_engine(
        f"postgresql+psycopg://{user}:{password}@{host}/{dbname}", echo=echo
    )


def initialize():
    engine = create_engine_from_env()
    Base.metadata.create_all(engine)
    index = Index(
        "text_node_index",
        StoredTextNode.embedding,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    index.create(engine)


def destroy():
    engine = create_engine_from_env()
    Base.metadata.drop_all(engine)


###############################################################################
#                              Data classes                                   #
###############################################################################


class Base(DeclarativeBase):
    pass


class WebPage(Base):
    __tablename__ = "web_pages"
    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str] = mapped_column(String(1024))
    html: Mapped[str] = mapped_column(Text())
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    def __repr__(self) -> str:
        return f"WebPage(id={self.id!r}, url={self.url!r}, length={len(self.html)!r})"


class StoredTextNode(Base):
    __tablename__ = "text_nodes"
    id: Mapped[uuid.UUID] = mapped_column(UUID, primary_key=True, default=uuid.uuid4)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[np.ndarray] = mapped_column(Vector(EMBEDDING_SIZE))
    meta: Mapped[dict] = mapped_column(JSON)
    start_char_idx: Mapped[Optional[int]] = mapped_column(Integer)
    end_char_idx: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    @staticmethod
    def from_text_node(node: TextNode):
        return StoredTextNode(
            id=node.id_,
            text=node.text,
            embedding=node.embedding,
            meta=node.extra_info,
            start_char_idx=node.start_char_idx,
            end_char_idx=node.end_char_idx,
        )

    def to_text_node(self):
        return TextNode(
            id_=str(self.id),
            text=self.text,
            embedding=self.embedding.tolist(),
            extra_info=self.meta,
            start_char_idx=self.start_char_idx,
            end_char_idx=self.end_char_idx,
        )
