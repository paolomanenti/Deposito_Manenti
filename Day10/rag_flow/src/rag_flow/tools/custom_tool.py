from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from rag_system import (SETTINGS,
                        get_embeddings,
                        make_retriever,
                        load_documents_from_folder,
                        load_or_build_vectorstore,
                        get_contexts_for_question,
                        make_retriever)

class FetchContextsInput(BaseModel):
    query: str = Field(..., description="The user question to fetch contexts for relevant information.")

class FetchContextsOutput(BaseTool):
    name: str = "context_retrieval_tool"
    description: str = "Fetch relevant contexts for a given question."
    input_model: Type[BaseModel] = FetchContextsInput

    def _run(self, query: str) -> str:
        embeddings = get_embeddings(SETTINGS)
        docs = load_documents_from_folder("src/rag_flow/data")
        vector_store = load_or_build_vectorstore(SETTINGS, embeddings, docs)
        retriever = make_retriever(vector_store=vector_store, settings=SETTINGS)
        contexts = get_contexts_for_question(retriever, query, SETTINGS.k)
        return "\n\n".join(contexts)
