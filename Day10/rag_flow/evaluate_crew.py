from typing import List
from ragas import EvaluationDataset, evaluate
from rag_system import (Settings,
                        get_contexts_for_question,
                        get_embeddings,
                        load_documents_from_folder,
                        load_or_build_vectorstore,
                        make_retriever,
                        get_llm)

from ragas.metrics import answer_correctness  # usa questa solo se hai ground_truth
from ragas.metrics import AnswerRelevancy  # pertinenza della risposta vs domanda
from ragas.metrics import context_precision  # "precision@k" sui chunk recuperati
from ragas.metrics import context_recall  # copertura dei chunk rilevanti
from ragas.metrics import faithfulness  # ancoraggio della risposta al contesto

from src.rag_flow.main import RagAgentFlow

SETTINGS = Settings()

def build_ragas_dataset(
    questions: List[str],
    retriever,
    agent_flow,
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
        agent_flow.input_query = q
        answer = agent_flow.kickoff()

        row = {
            # chiavi richieste da molte metriche Ragas
            "user_input": q,
            "retrieved_contexts": contexts,
            "response": agent_flow.state.answer,
        }
        if ground_truth and q in ground_truth:
            row["reference"] = ground_truth[q]

        dataset.append(row)
    return dataset

def evaluate_rag(settings: Settings, questions: List[str]) -> EvaluationDataset:
    """
    Esegue la pipeline RAG e valuta con Ragas.
    """
    embeddings = get_embeddings(settings)
    docs = load_documents_from_folder("src/rag_flow/data")
    vector_store = load_or_build_vectorstore(settings, embeddings, docs)
    retriever = make_retriever(vector_store, settings)
    llm = get_llm(settings)
    agent_flow = RagAgentFlow()

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
        agent_flow=agent_flow,
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

def main():
    settings = SETTINGS

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