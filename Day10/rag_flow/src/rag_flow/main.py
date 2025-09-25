from pydantic import BaseModel
from crewai.flow import Flow, listen, start, router, or_
from litellm import completion
from src.rag_flow.crews.rag_crew.rag_crew import RagCrew


class Config:
    model_emb: str = "text-embedding-ada-002"
    model: str = "azure/gpt-4o"
    path_save_dir: str = "./output/answer.md"
    topic: str = "Rivoluzione Francese"

config = Config()

########################################

class RagState(BaseModel):
    query: str = ""
    answer: str = ""


class RagAgentFlow(Flow[RagState]):

    input_query = None

    @start("Not Relevant")
    def start_flow(self):
        pass

    @listen(or_(start_flow, "Not Relevant"))
    def get_user_query(self):
        self.state.query = self.input_query

    @router(get_user_query)
    def evaluate_question(self):
        response = completion(
            model=config.model,
            messages=[
                {"role": "system",
                 "content": f"You are a judge and you have to check if the query is relevant to the following topic \'{config.topic}\'."
                 "Expected output: respond with a JSON format - not in markdown format - with a single field: {'is_relevant': True/False}"},
                {"role": "user",
                 "content": f"Is the following user question relevant to the context: \'{self.state.query}\'?"}
            ]
        )
        response_content = response.choices[0].message["content"]
        answer_json = eval(response_content)
        if answer_json['is_relevant']:
            return "Relevant"
        print("The question is not relevant to the topic.")
        return "Not Relevant"

    @listen("Relevant")
    def rag_answer(self):
        result = (
            RagCrew()
            .crew()
            .kickoff(inputs={"query": self.state.query})
        )
        self.state.answer = result.raw

    @listen(rag_answer)
    def save_answer(self):
        with open(config.path_save_dir, "w") as f:
            f.write(self.state.answer)
        

def kickoff():
    rag_agent_flow = RagAgentFlow()
    rag_agent_flow.input_query = input("Enter your question: ")
    rag_agent_flow.kickoff()


def plot():
    rag_agent_flow = RagAgentFlow()
    rag_agent_flow.plot()


if __name__ == "__main__":
    kickoff()