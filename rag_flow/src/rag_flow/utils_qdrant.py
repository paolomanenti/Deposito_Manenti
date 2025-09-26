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
    
    This class centralizes all configurable parameters, allowing easy tuning
    of the system's behavior without modifying the core logic.
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
    """Create an Azure OpenAI embeddings client.

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
    """Initialize a chat LLM bound to Azure OpenAI.

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
    """Load text and PDF files within a folder as LangChain documents.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing `.txt`, `.md`, or `.pdf` files.

    Returns
    -------
    list[Document]
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
    return QdrantClient(url=settings.qdrant_url)

def recreate_collection_for_rag(client: QdrantClient, settings: Settings, vector_size: int):
    """
    Create or recreate a Qdrant collection optimized for RAG (Retrieval-Augmented Generation).
    
    This function sets up a vector database collection with optimal configuration for
    semantic search, including HNSW indexing, payload indexing, and quantization.
    
    Args:
        client: Qdrant client instance for database operations
        settings: Configuration object containing collection parameters
        vector_size: Dimension of the embedding vectors (e.g., 384 for MiniLM-L6)
        
    Collection Architecture:
    - Vector storage: Dense vectors for semantic similarity search
    - Payload storage: Metadata and text content for retrieval
    - Indexing: HNSW for approximate nearest neighbor search
    - Quantization: Scalar quantization for memory optimization
        
    Distance Metric Selection:
    - Cosine distance: Normalized similarity, good for semantic embeddings
    - Alternatives: Euclidean (L2), Manhattan (L1), Dot product
    - Cosine preferred for normalized embeddings (sentence-transformers)
        
    HNSW Index Configuration:
    - m=32: Average connections per node (higher = better quality, more memory)
    - ef_construct=256: Search depth during construction (higher = better quality, slower build)
    - Trade-offs: Higher values improve recall but increase memory and build time
        
    Optimizer Configuration:
    - default_segment_number=2: Parallel processing segments
    - Benefits: Faster indexing, better resource utilization
    - Considerations: More segments = more memory overhead
        
    Quantization Strategy:
    - Scalar quantization: Reduces vector precision from float32 to int8
    - Memory savings: ~4x reduction in vector storage
    - Quality impact: Minimal impact on search accuracy
    - always_ram=False: Vectors stored on disk, loaded to RAM as needed
        
    Payload Indexing Strategy:
    - Text index: Full-text search capabilities (BM25 scoring)
    - Keyword indices: Fast exact matching and filtering
    - Performance: Significantly faster than unindexed field searches
        
    Collection Lifecycle:
    - recreate_collection: Drops existing collection and creates new one
    - Use case: Development/testing, major schema changes
    - Production: Consider using create_collection + update_collection_info
        
    Performance Considerations:
    - Build time: HNSW construction scales with collection size
    - Memory usage: Vectors loaded to RAM during search
    - Storage: Quantized vectors + payload data
    - Query latency: HNSW provides sub-millisecond search times
        
    Scaling Guidelines:
    - Small collections (<100K vectors): Current settings optimal
    - Medium collections (100K-1M vectors): Increase m to 48-64
    - Large collections (1M+ vectors): Consider multiple collections or sharding
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
    Usa l'indice full-text su 'text' per prefiltrare i punti che contengono parole chiave.
    Non restituisce uno score BM25: otteniamo un sottoinsieme di id da usare come boost.
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
    
    MMR balances relevance to the query with diversity among selected results,
    reducing redundancy and improving information coverage. This is particularly
    useful for RAG systems where diverse context provides better generation.
    
    Args:
        query_vec: Query embedding vector for relevance calculation
        candidates_vecs: List of candidate document embedding vectors
        k: Number of results to select
        lambda_mult: MMR parameter balancing relevance vs. diversity (0.0 to 1.0)
        
    Returns:
        List[int]: Indices of selected candidates in order of selection
        
    MMR Algorithm Overview:
    
    The algorithm iteratively selects candidates that maximize the MMR score:
    
    MMR_score(i) = λ × Relevance(i, query) - (1-λ) × max_similarity(i, selected)
    
    Where:
    - λ (lambda_mult): Weight for relevance vs. diversity
    - Relevance(i, query): Cosine similarity between candidate i and query
    - max_similarity(i, selected): Maximum similarity between candidate i and already selected items
        
    Algorithm Steps:
    
    1. INITIALIZATION:
       - Calculate relevance scores for all candidates vs. query
       - Select the highest-scoring candidate as the first result
       - Initialize selected and remaining candidate sets
        
    2. ITERATIVE SELECTION:
       - For each remaining position, calculate MMR score for all candidates
       - MMR score balances query relevance with diversity from selected items
       - Select candidate with highest MMR score
       - Update selected and remaining sets
        
    3. TERMINATION:
       - Continue until k candidates selected or no more candidates available
       - Return indices in selection order
        
    Mathematical Foundation:
    
    Cosine Similarity:
    - cos(a,b) = (a·b) / (||a|| × ||b||)
    - Range: [-1, 1] where 1 = identical, 0 = orthogonal, -1 = opposite
    - Normalized vectors typically have values in [0, 1] range
        
    MMR Score Calculation:
    - Relevance term: λ × cos(query, candidate)
    - Diversity term: (1-λ) × max(cos(candidate, selected_i))
    - Higher relevance increases score, higher similarity to selected decreases score
        
    Lambda Parameter Behavior:
    
    λ = 0.0 (Pure Diversity):
    - Only diversity matters, relevance ignored
    - Results may be irrelevant to query
    - Useful for exploratory search
        
    λ = 0.5 (Balanced):
    - Equal weight for relevance and diversity
    - Good compromise for general use
    - Moderate redundancy reduction
        
    λ = 0.6 (Current Setting):
    - Slight preference for relevance
    - Good diversity while maintaining relevance
    - Recommended for most RAG applications
        
    λ = 1.0 (Pure Relevance):
    - Only relevance matters, diversity ignored
    - Equivalent to simple top-K selection
    - May have redundant results
        
    Performance Characteristics:
    
    Time Complexity:
    - O(k × n) where k = results to select, n = total candidates
    - Each iteration processes all remaining candidates
    - Quadratic complexity in worst case (k ≈ n)
        
    Space Complexity:
    - O(n) for storing vectors and similarity scores
    - O(k) for selected indices
    - O(n) for remaining candidate set
        
    Memory Usage:
    - Vector storage: All candidate vectors loaded in memory
    - Similarity cache: Relevance scores computed once
    - Selection state: Small overhead for tracking
        
    Quality Metrics:
    
    Relevance Preservation:
    - Higher lambda values preserve more relevance
    - Lower lambda values may sacrifice relevance for diversity
    - Optimal balance depends on use case
        
    Diversity Improvement:
    - MMR significantly reduces redundancy compared to top-K
    - Diversity increases as lambda decreases
    - Measurable improvement in information coverage
        
    User Experience:
    - Less repetitive results
    - Better coverage of different aspects
    - More informative context for LLM generation
        
    Use Case Recommendations:
    
    Research & Exploration:
    - λ = 0.3-0.5: Maximize diversity for comprehensive understanding
    - Higher k values: More diverse perspectives
        
    Factual Queries:
    - λ = 0.7-0.9: Prioritize relevance for accurate information
    - Lower k values: Focus on most relevant results
        
    Technical Documentation:
    - λ = 0.5-0.7: Balance relevance with diverse technical perspectives
    - Moderate k values: Comprehensive technical coverage
        
    Conversational AI:
    - λ = 0.6-0.8: Good relevance with some diversity
    - Higher k values: Rich context for generation
        
    Tuning Guidelines:
    
    For Maximum Diversity:
    - Decrease lambda to 0.3-0.5
    - Increase k to 8-12 results
    - Monitor relevance quality
        
    For Maximum Relevance:
    - Increase lambda to 0.8-1.0
    - Decrease k to 3-6 results
    - Accept some redundancy
        
    For Balanced Results:
    - Use lambda = 0.6-0.7 (current setting)
    - Moderate k values (6-8)
    - Good compromise for most applications
        
    Implementation Notes:
    
    Numerical Stability:
    - Small epsilon (1e-12) added to prevent division by zero
    - Cosine similarity handles normalized vectors robustly
    - Float precision sufficient for similarity calculations
        
    Edge Cases:
    - Empty candidate list: Returns empty result
    - k > candidates: Returns all candidates
    - Single candidate: Returns that candidate regardless of lambda
        
    Optimization Opportunities:
    - Vector similarity could be pre-computed and cached
    - Parallel processing for large candidate sets
    - Early termination for very low diversity scores
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
    
    This function implements a sophisticated retrieval strategy that leverages both
    semantic understanding and traditional text search to provide high-quality,
    relevant results with minimal redundancy.
    
    Args:
        client: Qdrant client for database operations
        settings: Configuration object containing search parameters
        query: User's search query string
        embeddings: Embedding model for semantic search
        
    Returns:
        List[ScoredPoint]: Ranked list of relevant document chunks
        
    Hybrid Search Strategy Overview:
    
    1. SEMANTIC SEARCH (Vector Similarity):
       - Converts query to embedding vector
       - Performs approximate nearest neighbor search using HNSW index
       - Retrieves top_n_semantic candidates based on cosine similarity
       - Provides semantic understanding of query intent
        
    2. TEXT-BASED PREFILTERING:
       - Uses full-text search capabilities (BM25 scoring)
       - Identifies documents containing query keywords/phrases
       - Creates a set of text-relevant document IDs
       - Acts as a relevance filter for semantic results
        
    3. SCORE FUSION & NORMALIZATION:
       - Normalizes semantic scores to [0,1] range for fair comparison
       - Applies alpha weight to balance semantic vs. text relevance
       - Adds text_boost for results matching both criteria
       - Creates unified relevance scoring
        
    4. RESULT DIVERSIFICATION (Optional MMR):
       - Applies Maximal Marginal Relevance to reduce redundancy
       - Balances relevance with diversity using mmr_lambda parameter
       - Selects final_k results from top candidates
        
    Algorithm Flow:
    
    Phase 1: Semantic Retrieval
    - Query embedding generation
    - HNSW-based vector search
    - Score normalization for fusion
        
    Phase 2: Text Matching
    - Full-text search with MatchText filter
    - ID collection for hybrid scoring
    - Performance optimization with pagination
        
    Phase 3: Score Fusion
    - Linear combination of semantic and text scores
    - Boost application for hybrid matches
    - Ranking by fused scores
        
    Phase 4: Result Selection
    - Top-N selection or MMR diversification
    - Final result ordering and return
        
    Performance Characteristics:
    
    Time Complexity:
    - Semantic search: O(log n) with HNSW index
    - Text search: O(m) where m is text matches
    - Score fusion: O(k) where k is semantic candidates
    - MMR: O(k²) for diversity computation
        
    Memory Usage:
    - Vector storage: Quantized vectors in memory
    - Score storage: Temporary arrays for fusion
    - Result storage: Final selected points
        
    Quality Metrics:
    
    Recall (Completeness):
    - Semantic search: High recall for conceptual queries
    - Text search: High recall for keyword queries
    - Hybrid approach: Combines strengths of both
        
    Precision (Relevance):
    - Score fusion: Balances multiple relevance signals
    - Text boost: Rewards multi-criteria matches
    - MMR: Reduces redundant results
        
    Diversity:
    - MMR algorithm: Maximizes information coverage
    - Lambda parameter: Controls diversity vs. relevance trade-off
    - Result variety: Better user experience
        
    Tuning Guidelines:
    
    For High Precision:
    - Increase alpha (0.8-0.9): Prioritize semantic similarity
    - Increase text_boost (0.3-0.5): Reward text matches
    - Decrease mmr_lambda (0.7-0.9): Prioritize relevance
        
    For High Recall:
    - Increase top_n_semantic (50-100): More candidates
    - Increase top_n_text (150-200): More text matches
    - Decrease alpha (0.5-0.7): Balance search strategies
        
    For High Diversity:
    - Enable MMR (use_mmr=True)
    - Decrease mmr_lambda (0.3-0.6): Prioritize diversity
    - Increase final_k (8-12): More diverse results
        
    Use Case Optimizations:
    
    Technical Documentation:
    - High alpha (0.8-0.9): Semantic understanding critical
    - High text_boost (0.3-0.4): Technical terms important
    - MMR enabled: Diverse technical perspectives
        
    General Knowledge:
    - Balanced alpha (0.6-0.8): Both strategies valuable
    - Moderate text_boost (0.2-0.3): Balanced approach
    - MMR enabled: Comprehensive coverage
        
    Factual Queries:
    - High alpha (0.7-0.9): Semantic context important
    - Low text_boost (0.1-0.2): Facts over style
    - MMR optional: Precision over diversity
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
    blocks = []
    for p in points:
        pay = p.payload or {}
        blocks.append(pay.get('text',''))
    return blocks

def format_docs_for_prompt(points: Iterable[Any]) -> str:
    blocks = []
    for p in points:
        pay = p.payload or {}
        src = pay.get("source", "unknown")
        blocks.append(f"[source:{src}] {pay.get('text','')}")
    return "\n\n".join(blocks)

def build_rag_chain(llm):
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
    to intelligent question answering, showcasing the system's capabilities
    and providing a template for production deployment.
    
    Pipeline Overview:
    
    1. SYSTEM INITIALIZATION:
       - Load configuration settings
       - Initialize embedding model
       - Configure LLM (optional)
       - Establish database connection
        
    2. DOCUMENT PROCESSING:
       - Load or simulate document corpus
       - Split documents into manageable chunks
       - Generate vector embeddings for each chunk
        
    3. VECTOR DATABASE SETUP:
       - Create/configure Qdrant collection
       - Set up HNSW indexing and payload indices
       - Optimize for semantic search performance
        
    4. DATA INGESTION:
       - Store document chunks with metadata
       - Index vectors for fast retrieval
       - Ensure data consistency and availability
        
    5. INTELLIGENT RETRIEVAL:
       - Process user queries through hybrid search
       - Combine semantic and text-based matching
       - Apply MMR for result diversification
        
    6. CONTENT GENERATION:
       - Use LLM for intelligent answer generation
       - Fall back to content display if LLM unavailable
       - Provide source citations and context
        
    Performance Characteristics:
    
    Initialization Time:
    - Embedding model: 2-10 seconds (depends on model size)
    - LLM connection: 0.1-5 seconds (depends on service)
    - Database setup: 1-5 seconds (depends on collection size)
        
    Processing Time:
    - Document chunking: Linear with document count
    - Vector generation: Linear with chunk count
    - Database indexing: O(n log n) with HNSW construction
        
    Query Time:
    - Semantic search: Sub-millisecond with HNSW
    - Text search: Millisecond range with payload indices
    - Result fusion: Linear with candidate count
    - MMR diversification: Quadratic with candidate count
        
    Memory Usage:
    - Embedding model: 100MB-2GB (depends on model)
    - Vector storage: 4 bytes × dimensions × chunks (quantized)
    - Payload storage: Variable based on metadata size
    - LLM context: Depends on model and input size
        
    Scalability Considerations:
    
    Document Volume:
    - Small (<1K docs): Current settings optimal
    - Medium (1K-100K docs): Consider batch processing
    - Large (100K+ docs): Implement streaming ingestion
        
    Vector Dimensions:
    - 384 dimensions: Fast, memory-efficient, good quality
    - 768 dimensions: Higher quality, more memory, slower
    - 1024+ dimensions: Maximum quality, significant overhead
        
    Collection Management:
    - Single collection: Simple, good for small-medium datasets
    - Multiple collections: Better for large, diverse datasets
    - Sharding: Consider for very large datasets (>1M vectors)
        
    Error Handling Strategy:
    
    Graceful Degradation:
    - LLM failures: Fall back to content display
    - Database errors: Informative error messages
    - Network issues: Retry logic for transient failures
        
    Resource Management:
    - Memory monitoring: Prevent OOM conditions
    - Connection pooling: Efficient database usage
    - Cleanup: Proper resource deallocation
        
    Monitoring & Logging:
    - Performance metrics: Track response times
    - Error rates: Monitor system health
    - Usage patterns: Understand user behavior
        
    Production Deployment Considerations:
    
    Environment Configuration:
    - Use environment variables for sensitive data
    - Separate configs for dev/staging/production
    - Implement proper logging and monitoring
        
    Security:
    - API key management: Secure storage and rotation
    - Network security: HTTPS, firewall rules
    - Access control: User authentication and authorization
        
    Performance Optimization:
    - Caching: Redis for frequently accessed data
    - Load balancing: Distribute requests across instances
    - CDN: Static content delivery optimization
        
    Maintenance:
    - Regular backups: Database and configuration
    - Model updates: Periodic embedding model refresh
    - Performance tuning: Monitor and adjust parameters
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