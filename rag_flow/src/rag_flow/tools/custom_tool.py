from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from rag_flow.rag_system import (
    SETTINGS,
    get_embeddings,
    make_retriever,
    load_documents_from_folder,
    load_or_build_vectorstore,
    get_contexts_for_question,
    make_retriever,
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
        vector_store = load_or_build_vectorstore(SETTINGS, embeddings, docs)
        retriever = make_retriever(vector_store=vector_store, settings=SETTINGS)
        contexts = get_contexts_for_question(retriever, query, SETTINGS.k)
        return "\n\n".join(contexts)
