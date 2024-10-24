from dotenv import load_dotenv
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register
from langchain_openai import ChatOpenAI

_ = load_dotenv()

PHOENIX_COLLECTOR_ENDPOINT='http://0.0.0.0:6006'

tracer_provider = register(
  project_name="default",
  endpoint="http://0.0.0.0:6006/v1/traces"
)

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

def call_llm(query):
    response = llm.invoke(query)
    return response

my_query = "Whats aDS/CFT correspondence?"

print(call_llm(my_query))
