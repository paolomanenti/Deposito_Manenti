from typing import List
from ragas import EvaluationDataset, evaluate
from rag_system import (
    Settings,
    get_contexts_for_question,
    get_embeddings,
    load_documents_from_folder,
    load_or_build_vectorstore,
    make_retriever,
    get_llm,
)

from ragas.metrics import answer_correctness  # use only if you have ground_truth
from ragas.metrics import AnswerRelevancy  # answer relevancy vs question
from ragas.metrics import context_precision  # "precision@k" on retrieved chunks
from ragas.metrics import context_recall  # coverage of relevant chunks
from ragas.metrics import faithfulness  # answer grounded in context

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
    Build a dataset for Ragas evaluation by running the RAG pipeline for each question.

    Each row contains: question, retrieved contexts, answer, and optionally ground truth.

    :param questions: List of user questions to evaluate.
    :type questions: List[str]
    :param retriever: Retriever object to fetch relevant contexts.
    :param agent_flow: RagAgentFlow instance to generate answers.
    :param k: Number of top contexts to retrieve.
    :type k: int
    :param ground_truth: Optional dictionary mapping questions to reference answers.
    :type ground_truth: dict[str, str] or None
    :return: List of dictionaries, each representing a Ragas evaluation row.
    :rtype: list[dict]
    """
    dataset = []
    for q in questions:
        contexts = get_contexts_for_question(retriever, q, k)
        agent_flow.input_query = q
        answer = agent_flow.kickoff()

        row = {
            # keys required by many Ragas metrics
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
    Run the RAG pipeline and evaluate the results using Ragas metrics.

    This function builds the RAG dataset, selects appropriate metrics, and performs
    evaluation using the provided LLM and embeddings.

    :param settings: Settings object containing configuration for the RAG system.
    :type settings: Settings
    :param questions: List of user questions to evaluate.
    :type questions: List[str]
    :return: Pandas DataFrame with evaluation results.
    :rtype: pandas.DataFrame
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

    # Build dataset for Ragas (same top-k as your retriever)
    dataset = build_ragas_dataset(
        questions=questions,
        retriever=retriever,
        agent_flow=agent_flow,
        k=settings.k,
        ground_truth=ground_truth,  # remove if you do not want correctness
    )

    evaluation_dataset = EvaluationDataset.from_list(dataset)

    # Select metrics: answer_relevancy, faithfulness, context_recall, context_precision
    answer_relevancy = AnswerRelevancy(strictness=1)
    metrics = [answer_relevancy, faithfulness, context_recall, context_precision]
    # Add correctness only if all rows have ground_truth
    if all("ground_truth" in row for row in dataset):
        metrics.append(answer_correctness)

    # Run evaluation with your LLM and embeddings
    ragas_result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=llm,  # pass your LangChain LLM instance
        embeddings=embeddings,  # pass your LangChain embeddings instance
    )

    df = ragas_result.to_pandas()
    cols = ["user_input", "response", "faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    print("\n=== EXAMPLE DETAILS ===")
    print(df[cols].round(4).to_string(index=False))

    df.to_csv("./ragas_results.csv", index=False)
    print("Saved: ragas_results.csv")
    return df


def main():
    """
    Main entry point for RAG evaluation.

    Defines the questions, runs the evaluation, and prints the results.
    """
    settings = SETTINGS

    questions = [
        "Quando fu adottato il calendario rivoluzionario e come erano chiamati i mesi?",
        "Chi era Robespierre e quale ruolo ebbe nel Terrore?",
        "Quali furono le cause principali della Rivoluzione Francese?",
        "Quali furono le conseguenze della Rivoluzione Francese per la monarchia?"
    ]
    # Ragas
    df = evaluate_rag(settings, questions)
    print(df[['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']].round(4).to_string(index=False))


if __name__ == "__main__":
    main()