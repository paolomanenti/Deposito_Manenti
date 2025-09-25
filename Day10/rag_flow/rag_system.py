from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import faiss
# ricerca web
from ddgs import DDGS
from dotenv import load_dotenv
# Chat model init (provider-agnostic, qui puntiamo a LM Studio via OpenAI-compatible)
from langchain.chat_models import init_chat_model
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
# LangChain Core (prompt/chain)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (RunnableLambda, RunnableParallel,
                                      RunnablePassthrough)
from langchain_openai import AzureOpenAIEmbeddings
# Azure OpenAI client
from openai import AzureOpenAI
# pdf reader
from PyPDF2 import PdfReader
# --- RAGAS ---
from ragas import EvaluationDataset, evaluate
from ragas.metrics import answer_correctness  # usa questa solo se hai ground_truth
from ragas.metrics import AnswerRelevancy  # pertinenza della risposta vs domanda
from ragas.metrics import context_precision  # "precision@k" sui chunk recuperati
from ragas.metrics import context_recall  # copertura dei chunk rilevanti
from ragas.metrics import faithfulness  # ancoraggio della risposta al contesto
# =========================
# Configurazione
# =========================

load_dotenv()

@dataclass
class Settings:
    """Configuration container for the RAG pipeline.

    Attributes
    ----------
    persist_dir : str
        Directory where the FAISS index is persisted.
    chunk_size : int
        Maximum number of characters per text chunk.
    chunk_overlap : int
        Number of overlapping characters between consecutive chunks.
    search_type : str
        Retrieval search strategy, either "mmr" or "similarity".
    k : int
        Number of final retrieved documents.
    fetch_k : int
        Number of candidate documents for MMR retrieval.
    mmr_lambda : float
        Trade-off between diversity and similarity for MMR (0..1).
    endpoint : str | None
        Azure OpenAI endpoint URL.
    subscription_key : str | None
        Azure OpenAI API key.
    api_version : str
        Azure OpenAI API version.
    model_name_emb : str
        Logical embedding model name.
    deployment_emb : str
        Azure deployment name for embeddings.
    model_name_chat : str
        Logical chat model name.
    deployment_chat : str
        Azure deployment name for chat completions.
    """
    # Persistenza FAISS
    persist_dir: str = "faiss_index"
    # Text splitting
    chunk_size: int = 700
    chunk_overlap: int = 250
    # Retriever (MMR)
    search_type: str = "mmr"  # "mmr" o "similarity"
    k: int = 4  # risultati finali
    fetch_k: int = 20  # candidati iniziali (per MMR)
    mmr_lambda: float = 0.3  # 0 = diversificazione massima, 1 = pertinenza massima
    endpoint = os.getenv("AZURE_API_BASE")
    subscription_key = os.getenv("AZURE_API_KEY")
    # Embedding Azure
    api_version = "2024-12-01-preview"
    model_name_emb = "text-embedding-ada-002"
    deployment_emb = "text-embedding-ada-002"
    # Azure
    model_name_chat = "gpt-4o"
    deployment_chat = "gpt-4o"


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
            f"Imposta la variabile {settings.lmstudio_model_env} con il nome del modello caricato in LM Studio."
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
    docs = []
    for file_path in Path(folder_path).rglob("*"):
        if file_path.suffix.lower() in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            docs.append(
                Document(page_content=content, metadata={"source": file_path.name})
            )
        # Manage pdf files if needed
        elif file_path.suffix.lower() == ".pdf":
            reader = PdfReader(str(file_path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            docs.append(
                Document(page_content=text, metadata={"source": file_path.name})
            )
    return docs

def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    """Split documents into overlapping chunks for retrieval.

    Parameters
    ----------
    docs : list[Document]
        Input documents to split.
    settings : Settings
        Chunking configuration.

    Returns
    -------
    list[Document]
        Chunked documents suitable for vector indexing.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ": ",
            ", ",
            " ",
            "",  # fallback aggressivo
        ],
    )
    return splitter.split_documents(docs)

def build_faiss_vectorstore(
    chunks: List[Document], embeddings: AzureOpenAIEmbeddings, persist_dir: str
) -> FAISS:
    """Build a FAISS vector store from chunks and persist it to disk.

    Parameters
    ----------
    chunks : list[Document]
        Pre-split documents.
    embeddings : AzureOpenAIEmbeddings
        Embedding function for vectorization.
    persist_dir : str
        Directory to save the FAISS index and metadata.

    Returns
    -------
    FAISS
        Persisted FAISS vector store.
    """
    # Determina la dimensione dell'embedding
    vs = FAISS.from_documents(documents=chunks, embedding=embeddings)

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(persist_dir)
    return vs

def load_or_build_vectorstore(
    settings: Settings, embeddings: AzureOpenAIEmbeddings, docs: List[Document]
) -> FAISS:
    """Load a persisted FAISS index or build it from documents.

    Parameters
    ----------
    settings : Settings
        Global configuration (controls persist directory).
    embeddings : AzureOpenAIEmbeddings
        Embedding function used by the vector store.
    docs : list[Document]
        Source documents to index if no persisted index is found.

    Returns
    -------
    FAISS
        Loaded or freshly built FAISS vector store.
    """
    persist_path = Path(settings.persist_dir)
    index_file = persist_path / "index.faiss"
    meta_file = persist_path / "index.pkl"

    if index_file.exists() and meta_file.exists():
        # Dal 2024/2025 molte build richiedono il flag 'allow_dangerous_deserialization' per caricare pkl locali
        return FAISS.load_local(
            settings.persist_dir, embeddings, allow_dangerous_deserialization=True
        )

    chunks = split_documents(docs, settings)
    return build_faiss_vectorstore(chunks, embeddings, settings.persist_dir)

def make_retriever(vector_store: FAISS, settings: Settings):
    """Create a retriever from the vector store.

    Parameters
    ----------
    vector_store : FAISS
        Vector store that backs the retrieval.
    settings : Settings
        Retrieval configuration (search type and parameters).

    Returns
    -------
    BaseRetriever
        Configured retriever (MMR or similarity-based).
    """
    if settings.search_type == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": settings.k,
                "fetch_k": settings.fetch_k,
                "lambda_mult": settings.mmr_lambda,
            },
        )
    else:
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.k},
        )

def format_docs_for_prompt(docs: List[Document]) -> str:
    """Format retrieved documents into a prompt-ready string with sources.

    Parameters
    ----------
    docs : list[Document]
        Documents to be rendered into the context block.

    Returns
    -------
    str
        Concatenated context where each chunk is prefixed by its [source:].
    """
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc{i}")
        lines.append(f"[source:{src}] {d.page_content}")
    return "\n\n".join(lines)

def ddgs_search(query: str, max_results: int = 5) -> List[str]:
    """Perform a DuckDuckGo search and return textual snippets with links.

    Parameters
    ----------
    query : str
        Search query.
    max_results : int, default=5
        Maximum number of results to fetch.

    Returns
    -------
    list[str]
        Strings formatted as "[source:<url>] <snippet>".
    """
    results = []
    with DDGS(verify=False) as ddgs:
        search_results = ddgs.text(query, max_results=max_results)
        for r in search_results:
            results.append(f"[source:{r['href']}] {r['body']}")
    return results

def build_rag_chain(llm, retriever, web: bool = False):
    """Create a RAG chain: retrieval -> prompt -> LLM.

    Parameters
    ----------
    llm : BaseChatModel
        LangChain chat model instance.
    retriever : BaseRetriever
        Backend retriever to fetch context chunks.
    web : bool, default=False
        If True, augment internal knowledge with DuckDuckGo results.

    Returns
    -------
    Runnable
        LCEL runnable that maps a question to an answer string.
    """
    system_prompt = (
        "Sei un assistente esperto. Rispondi in INGLESE.\n"
        "Usa esclusivamente il CONTENUTO fornito nel contesto e le deduzioni LOGICHE che ne derivano direttamente. "
        "Non introdurre conoscenza esterna, non correggere o contraddire il contesto, anche se contiene errori.\n"
        "Quando fai una deduzione, assicurati che sia una conseguenza ragionevole e immediata delle affermazioni nel contesto "
        "Per le deduzioni, cita le fonti delle premesse su cui ti basi.\n"
        "Se l'informazione non è presente né logicamente deducibile dal contesto, scrivi: 'Non è presente nel contesto fornito.'\n"
        "Sii conciso, accurato e tecnicamente corretto."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Contesto (estratti selezionati):\n{context}\n\n"
                "Domanda:\n{question}\n\n"
                
            ),
        ]
    )


    # Struttura della catena RAG
    # combina contesto interno (retriever) e web (DDG) se web=True
    # e passa il contesto combinato al prompt

    if web:
        ddgs_runnable = RunnableLambda(lambda q: ddgs_search(q, max_results=5))

        context = (
            RunnableParallel(
                kb=retriever | format_docs_for_prompt,  # tua conoscenza interna
                web=ddgs_runnable,  # risultati DDG sulla stessa query
            )
            # Unisci le due parti in un unico blocco di contesto
            | RunnableLambda(
                lambda x: f"{x['kb']}\n{x['web']}"
            )  # inserire x['web']} per includere i risultati web
        )
    else:
        context = retriever | format_docs_for_prompt

    # LCEL: dict -> prompt -> llm -> parser
    chain = (
        {
            "context": context,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def get_contexts_for_question(retriever, question: str, k: int) -> List[str]:
    """Return the page contents of the top-k retrieved chunks.

    Parameters
    ----------
    retriever : BaseRetriever
        Retriever to query.
    question : str
        Natural language question used for retrieval.
    k : int
        Number of top documents to include.

    Returns
    -------
    list[str]
        Page contents of the retrieved chunks.
    """
    docs = retriever.invoke(question)[:k]
    return [d.page_content for d in docs]

def build_ragas_dataset(
    questions: List[str],
    retriever,
    chain,
    k: int,
    ground_truth: dict[str, str] | None = None,
):
    """Run the RAG pipeline and build a dataset compatible with Ragas.

    Each row contains: question, retrieved_contexts, response, and optionally
    reference (ground truth).

    Parameters
    ----------
    questions : list[str]
        Questions to evaluate.
    retriever : BaseRetriever
        Retriever used to fetch contexts.
    chain : Runnable
        RAG LCEL chain that generates answers.
    k : int
        Number of contexts to include in the dataset for each question.
    ground_truth : dict[str, str] | None
        Optional mapping from question to reference answer.

    Returns
    -------
    list[dict]
        Ragas-compatible dataset rows.
    """
    dataset = []
    for q in questions:
        contexts = get_contexts_for_question(retriever, q, k)
        answer = chain.invoke(q)

        row = {
            # chiavi richieste da molte metriche Ragas
            "user_input": q,
            "retrieved_contexts": contexts,
            "response": answer,
        }
        if ground_truth and q in ground_truth:
            row["reference"] = ground_truth[q]

        dataset.append(row)
    return dataset

def rag_answer(question: str, chain) -> str:
    """Execute the RAG chain for a single question.

    Parameters
    ----------
    question : str
        User question.
    chain : Runnable
        RAG runnable that returns a string answer.

    Returns
    -------
    str
        Model-generated answer grounded in the provided context.
    """
    return chain.invoke(question)

def execute_rag(settings: Settings, questions: List[str]) -> List[dict]:
    """Run the end-to-end RAG pipeline over a list of questions.

    Parameters
    ----------
    settings : Settings
        Global configuration for embeddings, LLM, and retrieval.
    questions : list[str]
        Questions to be answered by the RAG system.

    Returns
    -------
    list[dict]
        List of {"question": str, "answer": str} pairs.
    """
    embeddings = get_embeddings(settings)
    llm = get_llm(settings)

    # Usa il corpus simulato o carica da cartella
    # docs = simulate_corpus()
    #docs = load_documents_from_folder("Lezione_04/miniRag/data")
    docs = load_documents_from_folder("./data/EU_AI_ACT")
    vector_store = load_or_build_vectorstore(settings, embeddings, docs)
    retriever = make_retriever(vector_store, settings)
    chain = build_rag_chain(llm, retriever, web=False)

    results = []
    for q in questions:
        ans = rag_answer(q, chain)
        results.append({"question": q, "answer": ans})
    return results

def evaluate_rag(settings: Settings, questions: List[str]) -> EvaluationDataset:
    """Evaluate the RAG pipeline with Ragas metrics.

    Parameters
    ----------
    settings : Settings
        Global configuration for embeddings, LLM, and retrieval.
    questions : list[str]
        Questions used to build the evaluation dataset.

    Returns
    -------
    pandas.DataFrame
        DataFrame of per-sample metric results.
    """
    embeddings = get_embeddings(settings)
    llm = get_llm(settings)

    # Usa il corpus simulato o carica da cartella
    # docs = simulate_corpus()
    docs = load_documents_from_folder("src/rag_flow/data")
    vector_store = load_or_build_vectorstore(settings, embeddings, docs)
    retriever = make_retriever(vector_store, settings)
    chain = build_rag_chain(llm, retriever, web=False)

    ground_truth = {
        questions[0]: "Fu istituito dalla Convenzione il 6 ottobre 1793, con effetto retroattivo dal 22 settembre 1792 (1º Vendémiaire, An I). I mesi erano: Vendémiaire, Brumaire, Frimaire, Nivôse, Pluviôse, Ventôse, Germinal, Floréal, Prairial, Messidor, Thermidor, Fructidor.",
        questions[1]: "Maximilien Robespierre fu un leader giacobino che, tra la fine del 1793 e il 1794, dominò il Comitato di Salute Pubblica e guidò il Terrore (5 settembre 1793 – 27 luglio 1794); fu deposto e ghigliottinato il 28 luglio 1794 (10 Thermidor An II).",
        questions[2]: "Cause principali: crisi finanziaria dello Stato (debiti e sistema fiscale iniquo), cattivi raccolti e aumento dei prezzi, squilibri sociali e privilegi dei ceti alti, influenza dell’Illuminismo e dell’esempio americano, impasse politica di Luigi XVI.",
        questions[3]: "La monarchia fu abolita il 21 settembre 1792 e il giorno seguente fu proclamata la Repubblica; Luigi XVI venne giustiziato il 21 gennaio 1793: cadde l’Ancien Régime."
    }



    # Costruisci dataset per Ragas (stessi top-k del tuo retriever)
    dataset = build_ragas_dataset(
        questions=questions,
        retriever=retriever,
        chain=chain,
        k=settings.k,
        ground_truth=ground_truth,  # rimuovi se non vuoi correctness
    )

    evaluation_dataset = EvaluationDataset.from_list(dataset)

    # Scegli le metriche: answer_relevancy, faithfulness, context_recall, context_precision
    answer_relevancy = AnswerRelevancy(strictness=1)
    metrics = [answer_relevancy, faithfulness, context_recall, context_precision]
    # Aggiungi correctness solo se tutte le righe hanno ground_truth
    if all("ground_truth" in row for row in dataset):
        metrics.append(answer_correctness)

    # Esegui la valutazione con il TUO LLM e le TUE embeddings
    ragas_result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=llm,  # passa l'istanza LangChain del tuo LLM
        embeddings=embeddings,  # passa l'istanza LangChain delle tue embeddings
    )

    df = ragas_result.to_pandas()
    cols = ["user_input", "response", "faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    print("\n=== DETTAGLIO PER ESEMPIO ===")
    print(df[cols].round(4).to_string(index=False))

    df.to_csv("./ragas_results.csv", index=False)
    print("Salvato: ragas_results.csv")
    return df

def write_answers_to_file(results: List[dict], filename: str):
    """Write question/answer pairs to a text file.

    Parameters
    ----------
    results : list[dict]
        List of {"question": str, "answer": str} objects.
    filename : str
        Output file path.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for i, r in enumerate(results, start=1):
            f.write("=" * 80 + "\n")
            f.write(f"Question {i}\n")
            f.write(f"Q: {r['question']}\nA: {r['answer']}\n")


# ==========================
#     Valutazione RAGAS 
# ==========================


def main():
    """Entry point that runs a Ragas evaluation on predefined questions."""
    settings = SETTINGS

    # Random questions about docuements.md in data/<docs-topic> folder
    questions = ["Quando fu adottato il calendario rivoluzionario e come erano chiamati i mesi?",
                 "Chi era Robespierre e quale ruolo ebbe nel Terrore?",
                 "Quali furono le cause principali della Rivoluzione Francese?",
                 "Quali furono le conseguenze della Rivoluzione Francese per la monarchia?"
                 ]
    # Ragas
    df = evaluate_rag(settings, questions)
    print(df[['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']].round(4).to_string(index=False))    

if __name__ == "__main__":
    main()