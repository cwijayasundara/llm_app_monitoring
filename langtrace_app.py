import os
from dotenv import load_dotenv
from langtrace_python_sdk import langtrace
from langtrace_python_sdk.utils.with_root_span import with_langtrace_root_span
from langchain_openai import ChatOpenAI

_ = load_dotenv()

LANGTRACE_API_KEY = os.getenv("LANGTRACE_API_KEY")

# To run from local, use the LangTrace key from the local server
langtrace.init(
    api_key=LANGTRACE_API_KEY,
    api_host="http://localhost:3000/api/trace",
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

@with_langtrace_root_span()
def call_llm(query):
    response = llm.invoke(query)
    return response

my_query = "Whats aDS/CFT correspondence?"

print(call_llm(my_query))