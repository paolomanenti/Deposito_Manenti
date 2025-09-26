from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from rag_flow.utils_qdrant import (
    SETTINGS,
    get_embeddings,
    load_documents_from_folder,
    split_documents,
    get_qdrant_client,
    recreate_collection_for_rag,
    upsert_chunks,
    hybrid_search,
    format_docs_for_prompt
)


class FetchContextsInput(BaseModel):
    """
    Input model for the context retrieval tool.

    Attributes
    ----------
    query : str
        The user question to fetch contexts for relevant information.
    """
    query: str = Field(..., description="The user question to fetch contexts for relevant information.")

class FetchContextsOutput(BaseTool):
    """
    Tool for fetching relevant contexts for a given question.

    This tool loads documents, builds or loads a vector store, creates a retriever,
    and fetches the most relevant contexts for the provided query.

    Attributes
    ----------
    name : str
        Name of the tool.
    description : str
        Description of the tool's purpose.
    input_model : Type[BaseModel]
        The input model expected by this tool.
    """
    name: str = "context_retrieval_tool"
    description: str = "Fetch relevant contexts for a given question."
    input_model: Type[BaseModel] = FetchContextsInput

    def _run(self, query: str) -> str:
        """
        Run the context retrieval process for a given query.

        Parameters
        ----------
        query : str
            The user question to fetch relevant contexts for.

        Returns
        -------
        str
            A string containing the joined relevant contexts separated by double newlines.
        """
        embeddings = get_embeddings(SETTINGS)
        docs = load_documents_from_folder("src/rag_flow/data")
        chunks = split_documents(docs, SETTINGS)
        client = get_qdrant_client(SETTINGS)
        vector_size = len(embeddings.embed_query("test"))
        recreate_collection_for_rag(client, SETTINGS, vector_size)
        upsert_chunks(client, SETTINGS, chunks, embeddings)
        hits = hybrid_search(client, SETTINGS, query, embeddings)
        contexts = format_docs_for_prompt(hits)
        return contexts