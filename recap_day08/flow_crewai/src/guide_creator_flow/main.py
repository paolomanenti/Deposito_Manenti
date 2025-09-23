from pydantic import BaseModel

from crewai.flow import Flow, listen, start, router, or_

from guide_creator_flow.crews.crew_checker.crew_checker import CrewChecker
from guide_creator_flow.crews.crew_output.crew_output import CrewOutput


class QuestionState(BaseModel):
    question: str = ""
    ethic: bool | None = None
    answer: str = ""

INPUT = "Who is the highest man in the world?"

class EthicFlow(Flow[QuestionState]):

    @start()
    def define_user_input(self):
        self.state.question = input("Enter your question..")
        print("Starting flow with question:", self.state.question)
        
    
    @listen(or_(define_user_input, "failure"))
    def check_ethic(self):
        # call CrewChecker and set self.state.ethic
        result = (
            CrewChecker()
            .crew()
            .kickoff(inputs={"question": self.state.question})
        )
        print("Ethic checked:", result.pydantic.is_ethical, "\n")
        print("Explanation:", result.pydantic.explanation)
        self.state.ethic = result.pydantic.is_ethical
    
    @router(check_ethic)
    def checker(self):
        if self.state.ethic:
            return "success"
        return "failure"
    
    @listen("success")
    def answer(self):
        # call CrewOutput and set self.state.answer
        result = (
            CrewOutput()
            .crew()
            .kickoff(inputs={"question": self.state.question})
        )
        print("Answer generated:", result.raw)
        self.state.answer = result.raw

    
    @listen(answer)
    def save_answer(self):
        print("Saving answer")
        with open("output.md", "w") as f:
            f.write(self.state.answer)

    @listen("failure")
    def retry(self):
        print("The question was deemed unethical. Please provide a different question.")
    


def kickoff():
    ethic_flow = EthicFlow()
    ethic_flow.kickoff()


def plot():
    ethic_flow = EthicFlow()
    ethic_flow.plot("flow.png")


if __name__ == "__main__":
    #plot()
    kickoff()
