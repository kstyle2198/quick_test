from prompt import (DEFAULT_REPORT_STRUCTURE, 
                    report_planner_query_writer_instructions,
                    report_planner_instructions,
                    planner_message)
import os
from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(
#     model = "qwen/qwen3-14b:free", # "qwen/qwen3-14b:free", "qwen/qwen3-30b-a3b:free",
#     base_url="https://openrouter.ai/api/v1",
#     api_key="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",
#     )

from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0,max_tokens=3000,) # "gemma2-9b-it", qwen-qwq-32b

### Configuration ################################################
from dataclasses import dataclass, fields
from typing import Any, Optional, Dict 
from langchain_core.runnables import RunnableConfig

@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the chatbot."""
    # Common configuration
    report_structure: str = DEFAULT_REPORT_STRUCTURE # Defaults to the default report structure
    search_api_config: Optional[Dict[str, Any]] = None
    
    # Graph-specific configuration
    number_of_queries: int = 2 # Number of search queries to generate per iteration
    max_search_depth: int = 2 # Maximum number of reflection + search iterations
    planner_provider: str = "ollama"  # Defaults to Anthropic as provider
    planner_model: str = "qwen3:4b" # Defaults to claude-3-7-sonnet-latest
    planner_model_kwargs: Optional[Dict[str, Any]] = None # kwargs for planner_model
    writer_provider: str = "ollama" # Defaults to Anthropic as provider
    writer_model: str = "qwen3:4b" # Defaults to claude-3-5-sonnet-latest
    writer_model_kwargs: Optional[Dict[str, Any]] = None # kwargs for writer_model
    search_api_config: Optional[Dict[str, Any]] = None 
    
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})

#### Define States ##############################################
from typing import Annotated, List, TypedDict, Literal
from pydantic import BaseModel, Field
import operator

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")

class Queries(BaseModel):
    queries: List[SearchQuery] = Field(description="List of search queries.",)

class Section(BaseModel):
    name: str = Field(description="Name for this section of the report.",)
    description: str = Field(description="Brief overview of the main topics and concepts to be covered in this section.",)
    research: bool = Field(description="Whether to perform web research for this section of the report.")
    content: str = Field(description="The content of the section.")   

class Sections(BaseModel):
    sections: List[Section] = Field(description="Sections of the report.",)

class Feedback(BaseModel):
    grade: Literal["pass","fail"] = Field(
        description="Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail')."
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="List of follow-up search queries.",
    )

class ReportStateInput(TypedDict):
    topic: str # Report topic
    index_names:list
    k:int
    
class ReportStateOutput(TypedDict):
    final_report: str # Final report

class ReportState(TypedDict):
    topic: str # Report topic    
    feedback_on_report_plan: str # Feedback on the report plan
    sections: list[Section] # List of report sections 
    completed_sections: Annotated[list, operator.add] # Send() API key
    report_sections_from_research: str # String of any completed sections from research to write final sections
    final_report: str # Final report
    index_names:list
    k:int

### Init Elastic #########################################
from typing import Union, List
from langchain_ollama import OllamaEmbeddings
from langchain_elasticsearch import ElasticsearchStore, DenseVectorStrategy

def load_elastic_vectorstore(index_names: Union[str, List[str]]):
    # 단일 문자열인 경우 리스트로 변환
    if isinstance(index_names, str):
        index_names = [index_names]
    
    vector_store = ElasticsearchStore(
        index_name=index_names, 
        embedding=OllamaEmbeddings(
            base_url="http://localhost:11434", 
            model="bge-m3:latest"
        ), 
        es_url="http://localhost:9200",
        es_user="Kstyle",
        es_password="12345",
        )
    return vector_store

index_names = ["ship_safety"]
vector_store = load_elastic_vectorstore(index_names=index_names)


### Helper Fuction ########################################
def retrieve(queries:list):
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"fetch_k": 10, "k":5},)
    documents = []
    for query in queries:
        docs = retriever.invoke(query)
        documents.extend(docs)    
    return documents

def formatting_docs(docs:list):
    return "\n\n".join(doc.page_content for doc in docs)

from langchain_core.messages import HumanMessage, SystemMessage
# def generate_report_plan(topic:str, number_of_queries:int=2):

def generate_report_plan(state: ReportState, config: RunnableConfig):
    # Config
    configurable = Configuration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries

    # inputs
    topic = state["topic"]
    feedback = state.get("feedback_on_report_plan", None)

    # Process
    structured_llm = llm.with_structured_output(Queries)
    system_instructions_query = report_planner_query_writer_instructions.format(topic=topic, report_organization=DEFAULT_REPORT_STRUCTURE, number_of_queries=number_of_queries)
    results = structured_llm.invoke([SystemMessage(content=system_instructions_query), HumanMessage(content="보고서의 섹션을 계획하는 데 도움이 되는 검색 쿼리를 생성합니다.")])
    query_list = [query.search_query for query in results.queries]
    docs = retrieve(queries=query_list)
    source_str = formatting_docs(docs=docs)
    system_instructions_sections = report_planner_instructions.format(topic=topic, report_organization=DEFAULT_REPORT_STRUCTURE, context=source_str, feedback=feedback)
    structured_llm = llm.with_structured_output(Sections)
    report_sections = structured_llm.invoke([SystemMessage(content=system_instructions_sections), HumanMessage(content=planner_message)])
    sections = report_sections.sections
    return {"sections": sections}


from langgraph.constants import Send
from langgraph.types import interrupt, Command

def human_feedback(state: ReportState, config: RunnableConfig) -> Command[Literal["generate_report_plan"]]:
    """Get human feedback on the report plan and route to next steps.
    
    This node:
    1. Formats the current report plan for human review
    2. Gets feedback via an interrupt
    3. Routes to either:
       - Section writing if plan is approved
       - Plan regeneration if feedback is provided
    
    Args:
        state: Current graph state with sections to review
        config: Configuration for the workflow
        
    Returns:
        Command to either regenerate plan or start section writing
    """

    # Get sections
    topic = state["topic"]
    sections = state['sections']
    sections_str = "\n\n".join(
        f"Section: {section.name}\n"
        f"Description: {section.description}\n"
        f"Research needed: {'Yes' if section.research else 'No'}\n"
        for section in sections
    )

    # Get feedback on the report plan from interrupt
    interrupt_message = f"""Please provide feedback on the following report plan. 
                        \n\n{sections_str}\n
                        \nDoes the report plan meet your needs?\nPass 'true' to approve the report plan.\nOr, provide feedback to regenerate the report plan:"""
    
    feedback = interrupt(interrupt_message)

    # If the user approves the report plan, kick off section writing
    if isinstance(feedback, bool) and feedback is True:
        # Treat this as approve and kick off section writing
        return Command(goto=[
            Send("build_section_with_web_research", {"topic": topic, "section": s, "search_iterations": 0}) 
            for s in sections 
            if s.research
        ])
    
    # If the user provides feedback, regenerate the report plan 
    elif isinstance(feedback, str):
        # Treat this as feedback
        return Command(goto="generate_report_plan", 
                       update={"feedback_on_report_plan": feedback})
    else:
        raise TypeError(f"Interrupt value of type {type(feedback)} is not supported.")


from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=Configuration)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("human_feedback", human_feedback)

builder.add_edge(START, "generate_report_plan")
builder.add_edge("generate_report_plan", "human_feedback")
builder.add_edge("human_feedback", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

##################################################################################
##################################################################################
##################################################################################
if __name__ == "__main__":
    # inputs = {'topic': "최근 판례 동향을 고려한, 사고 유형별 중대재해 예방 대책"}
    # result = generate_report_plan(inputs, Configuration())
    # print(result)

    import uuid 

    thread = {"configurable": {"thread_id": str(uuid.uuid4()),
                               "max_search_depth": 1,
                               "report_structure": DEFAULT_REPORT_STRUCTURE,
                               }}


    topic = "최근 판례 동향을 고려한, 사고 유형별 중대재해 예방 대책"
    for event in graph.stream({"topic":topic,}, thread, stream_mode="updates"):
        if '__interrupt__' in event:
            interrupt_value = event['__interrupt__'][0].value
            print(interrupt_value)






