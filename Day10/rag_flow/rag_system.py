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
    """
    Restituisce un client di Azure configurato.
    """
    return AzureOpenAIEmbeddings(
        model=settings.deployment_emb,
        api_version=settings.api_version,
        azure_endpoint=settings.endpoint,
        api_key=settings.subscription_key,
    )

def get_llm(settings: Settings):
    """
    Inizializza un ChatModel puntando a LM Studio (OpenAI-compatible).
    Richiede:
      - OPENAI_BASE_URL (es. http://localhost:1234/v1)
      - OPENAI_API_KEY (placeholder qualsiasi, es. "not-needed")
      - LMSTUDIO_MODEL (nome del modello caricato in LM Studio)
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
    """
    Carica tutti i file di testo da una cartella come Documenti.
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
    """
    Applica uno splitting robusto ai documenti per ottimizzare il retrieval.
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
    """
    Costruisce da zero un FAISS index (IndexFlatL2) e lo salva su disco.
    """
    # Determina la dimensione dell'embedding
    vs = FAISS.from_documents(documents=chunks, embedding=embeddings)

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(persist_dir)
    return vs

def load_or_build_vectorstore(
    settings: Settings, embeddings: AzureOpenAIEmbeddings, docs: List[Document]
) -> FAISS:
    """
    Tenta il load di un indice FAISS persistente; se non esiste, lo costruisce e lo salva.
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
    """
    Configura il retriever. Con 'mmr' otteniamo risultati meno ridondanti e più coprenti.
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
    """
    Prepara il contesto per il prompt, includendo citazioni [source].
    """
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc{i}")
        lines.append(f"[source:{src}] {d.page_content}")
    return "\n\n".join(lines)

def ddgs_search(query: str, max_results: int = 5) -> List[str]:
    """
    Esegue una ricerca su DuckDuckGo e restituisce i risultati.
    """
    results = []
    with DDGS(verify=False) as ddgs:
        search_results = ddgs.text(query, max_results=max_results)
        for r in search_results:
            results.append(f"[source:{r['href']}] {r['body']}")
    return results

def build_rag_chain(llm, retriever, web: bool = False):
    """
    Costruisce la catena RAG (retrieval -> prompt -> LLM) con citazioni e regole anti-hallucination.
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
    """Ritorna i testi dei top-k documenti (chunk) usati come contesto."""
    docs = retriever.invoke(question)[:k]
    return [d.page_content for d in docs]

def build_ragas_dataset(
    questions: List[str],
    retriever,
    chain,
    k: int,
    ground_truth: dict[str, str] | None = None,
):
    """
    Esegue la pipeline RAG per ogni domanda e costruisce il dataset per Ragas.
    Ogni riga contiene: question, contexts, answer, (opzionale) ground_truth.
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
    """
    Esegue la catena RAG per una singola domanda.
    """
    return chain.invoke(question)

def execute_rag(settings: Settings, questions: List[str]) -> List[dict]:
    """
    Esegue l'intera pipeline RAG e restituisce le risposte.
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
    """
    Esegue la pipeline RAG e valuta con Ragas.
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
    """
    Scrive le risposte in un file di testo.
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