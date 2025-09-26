from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import AzureOpenAIEmbeddings
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain Core components for prompt/chain construction
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import init_chat_model

# Qdrant vector database client and models
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    PayloadSchemaType,
    FieldCondition,
    MatchValue,
    MatchText,
    Filter,
    SearchParams,
    PointStruct,
)

# =========================
# Configurazione
# =========================

load_dotenv(dotenv_path="C:\\Users\\PT776ZX\\OneDrive - EY\\Desktop\\Education\\AI Academy\\Deposito_Manenti\\rag_flow\\.env")


@dataclass
class Settings:
    """
    Comprehensive configuration settings for the RAG pipeline.

    Attributes
    ----------
    qdrant_url : str
        Qdrant server URL.
    collection : str
        Collection name for storing document chunks and vectors.
    hf_model_name : str
        HuggingFace sentence transformer model for generating embeddings.
    chunk_size : int
        Maximum number of characters per document chunk.
    chunk_overlap : int
        Number of characters to overlap between consecutive chunks.
    top_n_semantic : int
        Number of top semantic search candidates to retrieve initially.
    top_n_text : int
        Maximum number of text-based matches to consider for hybrid fusion.
    final_k : int
        Final number of results to return after all processing steps.
    alpha : float
        Weight for semantic similarity in hybrid score fusion.
    text_boost : float
        Additional score boost for results that match both semantic and text criteria.
    use_mmr : bool
        Whether to use MMR for result diversification and redundancy reduction.
    mmr_lambda : float
        MMR diversification parameter balancing relevance vs. diversity.
    lm_base_env : str
        Environment variable name for LLM service base URL.
    lm_key_env : str
        Environment variable name for LLM service API key.
    lm_model_env : str
        Environment variable name for the specific LLM model to use.
    deployment_emb : str
        Deployment name for the embedding model.
    api_version : str
        Azure API version.
    endpoint : str
        Azure API base endpoint.
    subscription_key : str
        Azure API subscription key.
    model_name_chat : str
        Name of the chat model deployed in Azure.
    deployment_chat : str
        Deployment name for the chat model.
    """
    
    # =========================
    # Qdrant Vector Database Configuration
    # =========================
    qdrant_url: str = "http://localhost:6333"
    """
    Qdrant server URL. 
    - Default: Local development instance
    - Production: Use your Qdrant cloud URL or server address
    - Alternative: Can be overridden via environment variable QDRANT_URL
    """
    
    collection: str = "rag_chunks"
    """
    Collection name for storing document chunks and vectors.
    - Naming convention: Use descriptive names like 'company_docs', 'research_papers'
    - Multiple collections: Can create separate collections for different document types
    - Cleanup: Old collections can be dropped and recreated for fresh indexing
    """
    
    # =========================
    # Embedding Model Configuration
    # =========================
    hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    """
    HuggingFace sentence transformer model for generating embeddings.
    
    Model Options & Trade-offs:
    - all-MiniLM-L6-v2: 384 dimensions, fast, good quality, balanced choice
    - all-MiniLM-L12-v2: 768 dimensions, slower, higher quality, better for complex queries
    - all-mpnet-base-v2: 768 dimensions, excellent quality, slower inference
    - paraphrase-multilingual-MiniLM-L12-v2: 768 dimensions, multilingual support
    
    Dimension Impact:
    - Lower dimensions (384): Faster search, less memory, slightly lower accuracy
    - Higher dimensions (768+): Better accuracy, slower search, more memory usage
    
    Performance Considerations:
    - L6 models: ~2-3x faster than L12 models
    - L12 models: ~10-15% better semantic understanding
    - Base models: Good balance between speed and quality
    """
    
    # =========================
    # Document Chunking Configuration
    # =========================
    chunk_size: int = 700
    """
    Maximum number of characters per document chunk.
    
    Chunk Size Trade-offs:
    - Small chunks (200-500): Better precision, more granular retrieval, higher storage overhead
    - Medium chunks (500-1000): Balanced precision and context, recommended for most use cases
    - Large chunks (1000+): Better context preservation, lower precision, fewer chunks to manage
    
    Optimal Sizing Guidelines:
    - Technical documents: 500-800 characters (preserve technical context)
    - General text: 700-1000 characters (good balance)
    - Conversational text: 300-600 characters (preserve dialogue flow)
    - Code/structured data: 200-500 characters (preserve logical units)
    
    Impact on Retrieval:
    - Smaller chunks: Higher recall, lower precision, more relevant snippets
    - Larger chunks: Lower recall, higher precision, more complete context
    """
    
    chunk_overlap: int = 120
    """
    Number of characters to overlap between consecutive chunks.
    
    Overlap Strategy:
    - No overlap (0): Clean separation, may miss context at boundaries
    - Small overlap (50-150): Preserves context, minimal redundancy
    - Large overlap (200+): Maximum context preservation, higher storage cost
    
    Optimal Overlap Guidelines:
    - Technical content: 100-200 characters (preserve technical terms)
    - General text: 100-150 characters (good balance)
    - Conversational: 50-100 characters (preserve dialogue context)
    - Code: 50-100 characters (preserve function boundaries)
    
    Storage Impact:
    - 0% overlap: Base storage requirement
    - 20% overlap: ~20% increase in storage
    - 50% overlap: ~50% increase in storage
    """
    
    # =========================
    # Hybrid Search Configuration
    # =========================
    top_n_semantic: int = 30
    """
    Number of top semantic search candidates to retrieve initially.
    
    Semantic Search Candidates:
    - Low values (10-20): Fast retrieval, may miss relevant results
    - Medium values (30-50): Good balance between speed and recall
    - High values (100+): Maximum recall, slower performance
    
    Performance Impact:
    - Retrieval time: Linear increase with candidate count
    - Memory usage: Linear increase with candidate count
    - Quality: Diminishing returns beyond 50-100 candidates
    
    Tuning Guidelines:
    - Small collections (<1000 docs): 20-30 candidates
    - Medium collections (1000-10000 docs): 30-50 candidates
    - Large collections (10000+ docs): 50-100 candidates
    """
    
    top_n_text: int = 100
    """
    Maximum number of text-based matches to consider for hybrid fusion.
    
    Text Search Scope:
    - Low values (50): Fast text filtering, may miss relevant matches
    - Medium values (100): Good balance between speed and coverage
    - High values (200+): Maximum text coverage, slower performance
    
    Hybrid Search Strategy:
    - Text search acts as a pre-filter for semantic results
    - Higher values improve the quality of text-semantic fusion
    - Optimal value depends on collection size and query complexity
    """
    
    final_k: int = 6
    """
    Final number of results to return after all processing steps.
    
    Result Count Considerations:
    - User experience: 3-5 results for simple queries, 5-10 for complex ones
    - Context window: Align with LLM context limits (e.g., 6-8 chunks for GPT-3.5)
    - Diversity: Higher values allow MMR to select more diverse results
    
    LLM Integration:
    - GPT-3.5: 6-8 chunks typically fit in context
    - GPT-4: 8-12 chunks can be processed
    - Claude: 6-10 chunks work well
    """
    
    alpha: float = 0.75
    """
    Weight for semantic similarity in hybrid score fusion (0.0 to 1.0).
    
    Alpha Parameter Behavior:
    - alpha = 0.0: Pure text-based ranking (BM25, keyword matching)
    - alpha = 0.5: Equal weight for semantic and text relevance
    - alpha = 0.75: Semantic similarity prioritized (current setting)
    - alpha = 1.0: Pure semantic ranking (cosine similarity only)
    
    Use Case Recommendations:
    - Technical queries: 0.7-0.9 (semantic understanding important)
    - Factual queries: 0.5-0.7 (balanced approach)
    - Keyword searches: 0.3-0.5 (text matching more important)
    - Conversational queries: 0.6-0.8 (semantic context matters)
    
    Tuning Strategy:
    - Start with 0.75 for general use
    - Increase if semantic results seem irrelevant
    - Decrease if text matching is too weak
    """
    
    text_boost: float = 0.20
    """
    Additional score boost for results that match both semantic and text criteria.
    
    Text Boost Mechanism:
    - Applied additively to fused scores
    - Encourages results that satisfy both search strategies
    - Helps surface highly relevant content that matches multiple criteria
    
    Boost Value Guidelines:
    - Low boost (0.1-0.2): Subtle preference for hybrid matches
    - Medium boost (0.2-0.4): Strong preference for hybrid matches
    - High boost (0.5+): Heavy preference, may dominate ranking
    
    Optimal Settings:
    - General use: 0.15-0.25
    - Technical content: 0.20-0.30
    - Factual queries: 0.10-0.20
    """
    
    # =========================
    # MMR (Maximal Marginal Relevance) Configuration
    # =========================
    use_mmr: bool = True
    """
    Whether to use MMR for result diversification and redundancy reduction.
    
    MMR Benefits:
    - Reduces redundant results with similar content
    - Improves coverage of different aspects of the query
    - Better user experience with diverse information
    
    MMR Trade-offs:
    - Slightly slower than simple top-K selection
    - May reduce absolute relevance scores
    - Better for exploratory queries, worse for specific fact retrieval
    
    Alternatives:
    - False: Simple top-K selection (faster, may have redundancy)
    - True: MMR diversification (slower, better diversity)
    """
    
    mmr_lambda: float = 0.6
    """
    MMR diversification parameter balancing relevance vs. diversity (0.0 to 1.0).
    
    Lambda Parameter Behavior:
    - lambda = 0.0: Pure diversity (ignore relevance, maximize difference)
    - lambda = 0.5: Balanced relevance and diversity
    - lambda = 0.6: Slight preference for relevance (current setting)
    - lambda = 1.0: Pure relevance (ignore diversity, top-K selection)
    
    Use Case Recommendations:
    - Research queries: 0.4-0.6 (diverse perspectives important)
    - Factual queries: 0.7-0.9 (relevance more important)
    - Exploratory queries: 0.3-0.5 (diversity valuable)
    - Specific searches: 0.8-1.0 (precision over diversity)
    
    Tuning Guidelines:
    - Start with 0.6 for general use
    - Decrease if results seem too similar
    - Increase if results seem too diverse
    """
    
    # =========================
    # LLM Configuration (Optional)
    # =========================
    lm_base_env: str = "OPENAI_BASE_URL"
    """
    Environment variable name for LLM service base URL.
    
    Supported Services:
    - OpenAI: https://api.openai.com/v1
    - LM Studio: http://localhost:1234/v1
    - Ollama: http://localhost:11434/v1
    - Custom API: Your endpoint URL
    
    Configuration Examples:
    - OpenAI: OPENAI_BASE_URL=https://api.openai.com/v1
    - LM Studio: OPENAI_BASE_URL=http://localhost:1234/v1
    - Azure OpenAI: OPENAI_BASE_URL=https://your-resource.openai.azure.com
    """
    
    lm_key_env: str = "OPENAI_API_KEY"
    """
    Environment variable name for LLM service API key.
    
    Security Notes:
    - Never hardcode API keys in source code
    - Use environment variables or secure secret management
    - Rotate keys regularly for production systems
    
    Configuration Examples:
    - OpenAI: OPENAI_API_KEY=sk-...
    - LM Studio: OPENAI_API_KEY=lm-studio (can be any value)
    - Azure: OPENAI_API_KEY=your-azure-key
    """
    
    lm_model_env: str = "LMSTUDIO_MODEL"
    """
    Environment variable name for the specific LLM model to use.
    
    Model Selection:
    - OpenAI: gpt-3.5-turbo, gpt-4, gpt-4-turbo
    - LM Studio: Any model name you've loaded
    - Ollama: llama2, codellama, mistral, etc.
    - Custom: Your model identifier
    
    Configuration Examples:
    - OpenAI: LMSTUDIO_MODEL=gpt-3.5-turbo
    - LM Studio: LMSTUDIO_MODEL=llama-2-7b-chat
    - Ollama: LMSTUDIO_MODEL=llama2:7b
    """

    deployment_emb: str = "text-embedding-ada-002"
    api_version = os.getenv("AZURE_API_VERSION")
    endpoint = os.getenv("AZURE_API_BASE")
    subscription_key = os.getenv("AZURE_API_KEY")
    model_name_chat = os.getenv("MODEL_NAME")
    deployment_chat: str = os.getenv("MODEL_NAME")

SETTINGS = Settings()

# =========================
# Componenti di base
# =========================

def get_embeddings(settings: Settings) -> AzureOpenAIEmbeddings:
    """
    Create an Azure OpenAI embeddings client.

    Parameters
    ----------
    settings : Settings
        Global configuration.

    Returns
    -------
    AzureOpenAIEmbeddings
        Configured embeddings client.
    """
    return AzureOpenAIEmbeddings(
        model=settings.deployment_emb,
        api_version=settings.api_version,
        azure_endpoint=settings.endpoint,
        api_key=settings.subscription_key,
    )

def get_llm(settings: Settings):
    """
    Initialize a chat LLM bound to Azure OpenAI.

    Parameters
    ----------
    settings : Settings
        Global configuration.

    Returns
    -------
    BaseChatModel
        LangChain chat model instance.

    Raises
    ------
    RuntimeError
        If mandatory Azure endpoint, key, or model settings are missing.
    """
    # base_url = os.getenv("OPENAI_BASE_URL")
    # api_key = os.getenv("OPENAI_API_KEY")
    # model_name = os.getenv(settings.lmstudio_model_env)

    if not settings.endpoint or not settings.subscription_key:
        raise RuntimeError("OPENAI_BASE_URL e OPENAI_API_KEY devono essere impostate")
    if not settings.model_name_chat:
        raise RuntimeError(
            f"Imposta la variabile {settings.deployment_chat} con il nome del modello caricato in Azure."
        )

    # model_provider="openai" perché l'endpoint è OpenAI-compatible
    return init_chat_model(
        settings.model_name_chat,
        model_provider="azure_openai",
        api_version=settings.api_version,
        azure_deployment=settings.deployment_chat,
        azure_endpoint=settings.endpoint,
        api_key=settings.subscription_key,
    )

def simulate_corpus() -> List[Document]:
    """
    Simulate a corpus of documents for testing purposes.

    Returns
    -------
    List[Document]
        A list of simulated documents.
    """

    docs = [
        Document(
            page_content=(
                "LangChain is a framework for building applications with Large Language Models. "
                "It provides chains, agents, prompt templates, memory, and many integrations."
            ),
            metadata={"id": "doc1", "source": "intro-langchain.md", "title": "Intro LangChain", "lang": "en"}
        ),
        Document(
            page_content=(
                "FAISS is a library for efficient similarity search of dense vectors. "
                "It supports both exact and approximate nearest neighbor search at scale."
            ),
            metadata={"id": "doc2", "source": "faiss-overview.md", "title": "FAISS Overview", "lang": "en"}
        ),
        Document(
            page_content=(
                "Sentence-transformers like all-MiniLM-L6-v2 produce 384-dimensional sentence embeddings "
                "for semantic search, clustering, and retrieval-augmented generation."
            ),
            metadata={"id": "doc3", "source": "embeddings-minilm.md", "title": "MiniLM Embeddings", "lang": "en"}
        ),
        Document(
            page_content=(
                "A typical RAG pipeline includes indexing (load, split, embed, store), retrieval, and generation. "
                "Retrieval selects the most relevant chunks, then the LLM answers grounded in those chunks."
            ),
            metadata={"id": "doc4", "source": "rag-pipeline.md", "title": "RAG Pipeline", "lang": "en"}
        ),
        Document(
            page_content=(
                "Maximal Marginal Relevance (MMR) trades off relevance and diversity to reduce redundancy "
                "and improve coverage of distinct aspects in retrieved chunks."
            ),
            metadata={"id": "doc5", "source": "retrieval-mmr.md", "title": "MMR Retrieval", "lang": "en"}
        ),
    ]
    return docs

def load_documents_from_folder(folder_path: str) -> List[Document]:
    """
    Load text and PDF files within a folder as LangChain documents.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing `.txt`, `.md`, or `.pdf` files.

    Returns
    -------
    List[Document]
        Documents with `page_content` and `source` metadata.
    """
    i = 1
    docs = []
    for file_path in Path(folder_path).rglob("*"):
        if file_path.suffix.lower() in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            docs.append(
                Document(page_content=content, metadata={"id":f"doc_{i}", "source": file_path.name, "lang":"ita", "title" : file_path.stem})
            )
        # Manage pdf files if needed
        elif file_path.suffix.lower() == ".pdf":
            reader = PdfReader(str(file_path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            docs.append(
                Document(page_content=text, metadata={"id":f"doc_{i}", "source": file_path.name, "lang":"ita", "title" : file_path.stem})
            )
        i =+ 1
    return docs

def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    """
    Split documents into smaller chunks based on the specified settings.

    Parameters
    ----------
    docs : List[Document]
        List of documents to split.
    settings : Settings
        Configuration settings for chunking.

    Returns
    -------
    List[Document]
        List of document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""],
    )
    return splitter.split_documents(docs)

# =========================
# Qdrant: creazione collection + indici
# =========================

def get_qdrant_client(settings: Settings) -> QdrantClient:
    """
    Initialize a Qdrant client.

    Parameters
    ----------
    settings : Settings
        Global configuration.

    Returns
    -------
    QdrantClient
        Configured Qdrant client.
    """
    return QdrantClient(url=settings.qdrant_url)

def recreate_collection_for_rag(client: QdrantClient, settings: Settings, vector_size: int):
    """
    Create or recreate a Qdrant collection optimized for RAG (Retrieval-Augmented Generation).

    Parameters
    ----------
    client : QdrantClient
        Qdrant client instance for database operations.
    settings : Settings
        Configuration object containing collection parameters.
    vector_size : int
        Dimension of the embedding vectors (e.g., 384 for MiniLM-L6).
    """
    client.recreate_collection(
        collection_name=settings.collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(
            m=32,             # grado medio del grafo HNSW (maggiore = più memoria/qualità)
            ef_construct=256  # ampiezza lista candidati in fase costruzione (qualità/tempo build)
        ),
        optimizers_config=OptimizersConfigDiff(
            default_segment_number=2  # parallelismo/segmentazione iniziale
        ),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(type="int8", always_ram=False)  # on-disk quantization dei vettori
        ),
    )

    # Indice full-text sul campo 'text' per filtri MatchText
    client.create_payload_index(
        collection_name=settings.collection,
        field_name="text",
        field_schema=PayloadSchemaType.TEXT
    )

    # Indici keyword per filtri esatti / velocità nei filtri
    for key in ["doc_id", "source", "title", "lang"]:
        client.create_payload_index(
            collection_name=settings.collection,
            field_name=key,
            field_schema=PayloadSchemaType.KEYWORD
        )

# =========================
# Ingest: chunk -> embed -> upsert
# =========================

def build_points(chunks: List[Document], embeds: List[List[float]]) -> List[PointStruct]:
    """
    Build Qdrant points from document chunks and their embeddings.

    Parameters
    ----------
    chunks : List[Document]
        List of document chunks.
    embeds : List[List[float]]
        List of embedding vectors corresponding to the chunks.

    Returns
    -------
    List[PointStruct]
        List of Qdrant points.
    """
    pts: List[PointStruct] = []
    for i, (doc, vec) in enumerate(zip(chunks, embeds), start=1):
        payload = {
            "doc_id": doc.metadata.get("id"),
            "source": doc.metadata.get("source"),
            "title": doc.metadata.get("title"),
            "lang": doc.metadata.get("lang", "en"),
            "text": doc.page_content,
            "chunk_id": i - 1
        }
        pts.append(PointStruct(id=i, vector=vec, payload=payload))
    return pts

def upsert_chunks(client: QdrantClient, settings: Settings, chunks: List[Document], embeddings: AzureOpenAIEmbeddings):
    """
    Upsert document chunks into the Qdrant collection.

    Parameters
    ----------
    client : QdrantClient
        Qdrant client instance.
    settings : Settings
        Configuration settings.
    chunks : List[Document]
        List of document chunks.
    embeddings : AzureOpenAIEmbeddings
        Embedding model for generating vector representations.
    """
    vecs = embeddings.embed_documents([c.page_content for c in chunks])
    points = build_points(chunks, vecs)
    client.upsert(collection_name=settings.collection, points=points, wait=True)

# =========================
# Ricerca: semantica / testuale / ibrida
# =========================

def qdrant_semantic_search(
    client: QdrantClient,
    settings: Settings,
    query: str,
    embeddings: AzureOpenAIEmbeddings,
    limit: int,
    with_vectors: bool = False
):
    """
    Perform semantic search in the Qdrant collection.

    Parameters
    ----------
    client : QdrantClient
        Qdrant client instance.
    settings : Settings
        Configuration settings.
    query : str
        Search query.
    embeddings : AzureOpenAIEmbeddings
        Embedding model for generating query vectors.
    limit : int
        Maximum number of results to retrieve.
    with_vectors : bool, optional
        Whether to include vectors in the results, by default False.

    Returns
    -------
    List[ScoredPoint]
        List of scored points from the Qdrant collection.
    """
    qv = embeddings.embed_query(query)
    res = client.query_points(
        collection_name=settings.collection,
        query=qv,
        limit=limit,
        with_payload=True,
        with_vectors=with_vectors,
        search_params=SearchParams(
            hnsw_ef=256,  # ampiezza lista in fase di ricerca (recall/latency)
            exact=False   # True = ricerca esatta (lenta); False = ANN HNSW
        ),
    )
    return res.points

def qdrant_text_prefilter_ids(
    client: QdrantClient,
    settings: Settings,
    query: str,
    max_hits: int
) -> List[int]:
    """
    Use the full-text index on 'text' to prefilter points containing keywords.

    Parameters
    ----------
    client : QdrantClient
        Qdrant client instance.
    settings : Settings
        Configuration settings.
    query : str
        Search query.
    max_hits : int
        Maximum number of text matches to retrieve.

    Returns
    -------
    List[int]
        List of IDs of matching points.
    """
    # Scroll con filtro MatchText per ottenere id dei match testuali
    # (nota: scroll è paginato; qui prendiamo solo i primi max_hits per semplicità)
    matched_ids: List[int] = []
    next_page = None
    while True:
        points, next_page = client.scroll(
            collection_name=settings.collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="text", match=MatchText(text=query))]
            ),
            limit=min(256, max_hits - len(matched_ids)),
            offset=next_page,
            with_payload=False,
            with_vectors=False,
        )
        matched_ids.extend([p.id for p in points])
        if not next_page or len(matched_ids) >= max_hits:
            break
    return matched_ids

def mmr_select(
    query_vec: List[float],
    candidates_vecs: List[List[float]],
    k: int,
    lambda_mult: float
) -> List[int]:
    """
    Select diverse results using Maximal Marginal Relevance (MMR) algorithm.

    Parameters
    ----------
    query_vec : List[float]
        Query embedding vector for relevance calculation.
    candidates_vecs : List[List[float]]
        List of candidate document embedding vectors.
    k : int
        Number of results to select.
    lambda_mult : float
        MMR parameter balancing relevance vs. diversity (0.0 to 1.0).

    Returns
    -------
    List[int]
        Indices of selected candidates in order of selection.
    """
    import numpy as np
    V = np.array(candidates_vecs, dtype=float)
    q = np.array(query_vec, dtype=float)

    def cos(a, b):
        na = (a @ a) ** 0.5 + 1e-12
        nb = (b @ b) ** 0.5 + 1e-12
        return float((a @ b) / (na * nb))

    sims = [cos(v, q) for v in V]
    selected: List[int] = []
    remaining = set(range(len(V)))

    while len(selected) < min(k, len(V)):
        if not selected:
            # pick the highest similarity first
            best = max(remaining, key=lambda i: sims[i])
            selected.append(best)
            remaining.remove(best)
            continue
        best_idx = None
        best_score = -1e9
        for i in remaining:
            max_div = max([cos(V[i], V[j]) for j in selected]) if selected else 0.0
            score = lambda_mult * sims[i] - (1 - lambda_mult) * max_div
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected

def hybrid_search(
    client: QdrantClient,
    settings: Settings,
    query: str,
    embeddings: AzureOpenAIEmbeddings
):
    """
    Perform hybrid search combining semantic similarity and text-based matching.

    Parameters
    ----------
    client : QdrantClient
        Qdrant client for database operations.
    settings : Settings
        Configuration object containing search parameters.
    query : str
        User's search query string.
    embeddings : AzureOpenAIEmbeddings
        Embedding model for semantic search.

    Returns
    -------
    List[ScoredPoint]
        Ranked list of relevant document chunks.
    """
    # (1) semantica
    sem = qdrant_semantic_search(
        client, settings, query, embeddings,
        limit=settings.top_n_semantic, with_vectors=True
    )
    if not sem:
        return []

    # (2) full-text prefilter (id)
    text_ids = set(qdrant_text_prefilter_ids(client, settings, query, settings.top_n_text))

    # Normalizzazione score semantici per fusione
    scores = [p.score for p in sem]
    smin, smax = min(scores), max(scores)
    def norm(x):  # robusto al caso smin==smax
        return 1.0 if smax == smin else (x - smin) / (smax - smin)

    # (3) fusione con boost testuale
    fused: List[Tuple[int, float, Any]] = []  # (idx, fused_score, point)
    for idx, p in enumerate(sem):
        base = norm(p.score)                    # [0..1]
        fuse = settings.alpha * base
        if p.id in text_ids:
            fuse += settings.text_boost         # boost additivo
        fused.append((idx, fuse, p))

    # ordina per fused_score desc
    fused.sort(key=lambda t: t[1], reverse=True)

    # MMR opzionale per diversificare i top-K
    if settings.use_mmr:
        qv = embeddings.embed_query(query)
        # prendiamo i primi N dopo fusione (es. 30) e poi MMR per final_k
        N = min(len(fused), max(settings.final_k * 5, settings.final_k))
        cut = fused[:N]
        vecs = [sem[i].vector for i, _, _ in cut]
        mmr_idx = mmr_select(qv, vecs, settings.final_k, settings.mmr_lambda)
        picked = [cut[i][2] for i in mmr_idx]
        return picked

    # altrimenti, prendi i primi final_k dopo fusione
    return [p for _, _, p in fused[:settings.final_k]]

# =========================
# Prompt/Chain per generazione con citazioni
# =========================

def format_docs_for_ragas(points: Iterable[Any]) -> List[str]:
    """
    Format documents for RAGAS evaluation.

    Parameters
    ----------
    points : Iterable[Any]
        Points retrieved from the Qdrant collection.

    Returns
    -------
    List[str]
        List of formatted document texts.
    """
    blocks = []
    for p in points:
        pay = p.payload or {}
        blocks.append(pay.get('text',''))
    return blocks

def format_docs_for_prompt(points: Iterable[Any]) -> str:
    """
    Format documents for inclusion in a prompt.

    Parameters
    ----------
    points : Iterable[Any]
        Points retrieved from the Qdrant collection.

    Returns
    -------
    str
        Formatted string of document texts with sources.
    """
    blocks = []
    for p in points:
        pay = p.payload or {}
        src = pay.get("source", "unknown")
        blocks.append(f"[source:{src}] {pay.get('text','')}")
    return "\n\n".join(blocks)

def build_rag_chain(llm):
    """
    Build a RAG chain for generating answers with citations.

    Parameters
    ----------
    llm : BaseChatModel
        Language model instance.

    Returns
    -------
    Runnable
        Configured RAG chain.
    """
    system_prompt = (
        "Sei un assistente tecnico. Rispondi in italiano, conciso e accurato. "
        "Usa ESCLUSIVAMENTE le informazioni presenti nel CONTENUTO. "
        "Se non è presente, dichiara: 'Non è presente nel contesto fornito.' "
        "Cita sempre le fonti nel formato [source:FILE]."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",
         "Domanda:\n{question}\n\n"
         "CONTENUTO:\n{context}\n\n"
         "Istruzioni:\n"
         "1) Risposta basata solo sul contenuto.\n"
         "2) Includi citazioni [source:...].\n"
         "3) Niente invenzioni.")
    ])

    chain = (
        {
            "context": RunnablePassthrough(),  # stringa già formattata
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# =========================
# Main end-to-end demo
# =========================

def main():
    """
    Main execution function demonstrating the complete RAG pipeline.

    This function orchestrates the entire RAG workflow from document ingestion
    to intelligent question answering.
    """
    s = SETTINGS
    embeddings = get_embeddings(s)
    llm = get_llm(s)  # opzionale

    # 1) Client Qdrant
    client = get_qdrant_client(s)

    # 2) Dati -> chunk
    docs = simulate_corpus()
    chunks = split_documents(docs, s)

    # 3) Crea (o ricrea) collection
    if s.deployment_emb == "text-embedding-ada-002":
        emb_dimensions = 1536

    assert len(embeddings.embed_query("Ciao sono Claudia")) == emb_dimensions, "Dimensione embedding errata, run una query e check lunghezza vettore."

    vector_size = emb_dimensions
    recreate_collection_for_rag(client, s, vector_size)

    # 4) Upsert chunks
    upsert_chunks(client, s, chunks, embeddings)

    # 5) Query ibrida
    questions = [
        "Cos'è una pipeline RAG e quali sono le sue fasi?",
        "A cosa serve FAISS e che caratteristiche offre?",
        "Che cos'è MMR e perché riduce la ridondanza?",
        "Qual è la dimensione degli embedding di all-MiniLM-L6-v2?",
    ]

    for q in questions:
        hits = hybrid_search(client, s, q, embeddings)
        print("=" * 80)
        print("Q:", q)
        if not hits:
            print("Nessun risultato.")
            continue

        # Mostra id/score di debug
        for p in hits:
            print(f"- id={p.id} score={p.score:.4f} src={p.payload.get('source')}")

        # Se LLM configurato: genera
        if llm:
            try:
                ctx = format_docs_for_prompt(hits)
                chain = build_rag_chain(llm)
                answer = chain.invoke({"question": q, "context": ctx})
                print("\n", answer, "\n")
            except Exception as e:
                print(f"\nLLM generation failed: {e}")
                print("Falling back to content display...")
                print("\nContenuto recuperato:\n")
                print(format_docs_for_prompt(hits))
                print()
        else:
            # Fallback: stampa i chunk per ispezione
            print("\nContenuto recuperato:\n")
            print(format_docs_for_prompt(hits))
            print()

if __name__ == "__main__":
    main()