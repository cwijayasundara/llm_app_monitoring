import os
import bs4
import warnings
from dotenv import load_dotenv
from langtrace_python_sdk import langtrace
from langtrace_python_sdk.utils.with_root_span import with_langtrace_root_span
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

warnings.filterwarnings('ignore')
_ = load_dotenv()

LANGTRACE_API_KEY = os.getenv("LANGTRACE_API_KEY")

langtrace.init(
    api_key=LANGTRACE_API_KEY,
    api_host="http://localhost:3000/api/trace",
)

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

prompt = hub.pull("rlm/rag-prompt")

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@with_langtrace_root_span()
def ingest_documents(doc_loader):
    docs = doc_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits,embedding=OpenAIEmbeddings())
    vec_retriever = vectorstore.as_retriever()
    return vec_retriever

vector_retriever = ingest_documents(loader)

@with_langtrace_root_span()
def query_vector_db(query, retriever):
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke(query)
    return response

question = "What is Task Decomposition?"

print(query_vector_db(question, vector_retriever))

