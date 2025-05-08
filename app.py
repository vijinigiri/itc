import streamlit as st
from tavily import TavilyClient
from langchain.schema import Document
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# Custom embedding class
class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, convert_to_numpy=True).tolist()

@st.cache_resource
def setup_rag_pipeline():
    # Step 1: Extract PDF content using Tavily
    client = TavilyClient(api_key="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")
    urls = [
        "https://www.itcportal.com/about-itc/shareholder-value/annual-reports/itc-annual-report-2023/pdf/ITC-Report-and-Accounts-2023.pdf",
        "https://www.itcportal.com/about-itc/shareholder-value/annual-reports/itc-annual-report-2024/pdf/ITC-Report-and-Accounts-2024.pdf"
    ]
    response = client.extract(urls=urls, include_images=False, extract_depth="advanced")

    docs = []
    for r in response["results"]:
        text = r.get("raw_content", "")
        if text:
            docs.append(Document(page_content=text, metadata={"source": r.get("url")}))

    # Step 2: Add Excel data
    excel_path = "aaa.xlsx"  # Make sure this file is in your root directory
    if os.path.exists(excel_path):
        excel_df = pd.read_excel(excel_path)
        for _, row in excel_df.iterrows():
            row_dict = row.astype(str).to_dict()
            row_text = "\n".join([f"{k}: {v}" for k, v in row_dict.items()])
            docs.append(Document(page_content=row_text, metadata={"source": "excel"}))

    # Step 3: Prepare embeddings
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    embedding = LocalSentenceTransformerEmbeddings()
    vectordb = Chroma.from_documents(split_docs, embedding=embedding)

    # Step 4: LLM
    genai.configure(api_key="YOUR_GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.2)

    # Step 5: Prompt + RAG chain
    retriever = vectordb.as_retriever()
    prompt_template = """
    Answer the question using the context below. If the answer is not in the context, say "I don't know".

    Context: {context}

    Question: {question}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    rag_chain = (
        {"context": retriever | RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# Streamlit UI
st.set_page_config(page_title="ITC Financial RAG", layout="centered")
st.title("📊 ITC Financial Assistant (Gemini RAG)")

rag_chain = setup_rag_pipeline()

question = st.text_input("Ask a financial question about ITC:", placeholder="e.g. What was ITC's net profit in 2024?")

if question:
    with st.spinner("Thinking..."):
        answer = rag_chain.invoke(question)
        st.success(answer)
