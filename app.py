from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import re
import tempfile
from typing import Dict, List
import streamlit as st
import json
from dotenv import load_dotenv

from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import CrossEncoder
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams

import uuid


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
import pdfplumber
from langchain.docstore.document import Document

from langchain_community.document_loaders import PyPDFLoader
from langchain_together import ChatTogether
from langchain.memory import ConversationBufferMemory

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
# MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
LOCAL_MODEL_PATH = "/Users/gautamnaik/models/Mistral-7B-Instruct-v0.3.Q8_0.gguf"



if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



def init_llm():
    return ChatTogether(
        api_key=os.getenv("TOGETHER_AI_API_KEY"),
        temperature=0.0,
        model=MODEL_NAME
    )


def get_local_llm():
    llm = LlamaCpp(
        model_path=LOCAL_MODEL_PATH,
        n_ctx=2048,
        n_threads=6,
        n_gpu_layers=32,
        temperature=0.7
    )
    return llm


def get_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )


@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./models/embeddings"  
    )



@st.cache_resource
def get_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") 


def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )

def normalize_section_headers(text: str) -> str:
    text = re.sub(r'\n(?=\d+\.\s+)', '\n\n', text) 
    text = re.sub(r'\n(?=(Introduction|Abstract|Conclusion|References|Appendix))', r'\n\n', text, flags=re.IGNORECASE)
    return text

def clean_text(text: str) -> str:
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'page \d+', '', text, flags=re.IGNORECASE)
    return text.strip()


def process_multiple_pdfs(files) -> List[Document]:
    all_docs = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

            try:
                with pdfplumber.open(temp_file_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        raw_text = page.extract_text()
                        if not raw_text:
                            continue

            
                        lines = raw_text.split('\n')
                        filtered_lines = [
                            line for line in lines
                            if not re.match(r"^(page \d+|header|footer)", line.strip(), flags=re.IGNORECASE)
                        ]
                        cleaned_text = clean_text("\n".join(filtered_lines))

                        if cleaned_text:
                            doc = Document(
                                page_content=cleaned_text,
                                metadata={
                                    "doc_name": file.name.replace(".pdf", ""),
                                    "page": i + 1
                                }
                            )
                            all_docs.append(doc)

                        tables = page.extract_tables()
                        for table in tables:
                            table_str = '\n'.join([' | '.join(row) for row in table if any(row)])
                            table_doc = Document(
                                page_content=f"Table from page {i + 1}:\n{table_str}",
                                metadata={
                                    "doc_name": file.name.replace(".pdf", ""),
                                    "page": i + 1,
                                    "type": "table"
                                }
                            )
                            all_docs.append(table_doc)

            finally:
                os.remove(temp_file_path)

    return all_docs

    


def create_qa_chain(documents, embedding_model):
    text_splitter = get_text_splitter()

    for doc in documents:
        doc.page_content = normalize_section_headers(doc.page_content)

    splits = text_splitter.split_documents(documents)

    if not splits:
        return None

    # client = QdrantClient(":memory:")

    # client.create_collection(
    #     collection_name="demo_collection",
    #     vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    # )

    # vectorstore = QdrantVectorStore(
    #     client=client,
    #     collection_name="demo_collection",
    #     embedding=embedding_model,
    # )

    # vectorstore.add_documents(splits)

        
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embedding_model
    )

    memory = get_memory()

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=get_local_llm(),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True,
        output_key="answer",
    )

    return qa_chain


def format_chat_history(chat_history: list) -> str:
    return "\n".join([f"User: {q}\nAssistant: {a}" for q, a, _ in chat_history])


def rerank_results(query):

    docs = st.session_state.qa_chain.retriever.get_relevant_documents(query)

    cross_encoder = get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    ranked_docs = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    top_docs = ranked_docs[:3]
    context = "\n\n".join([doc.page_content for doc in top_docs])
    prompt = f"""You are a helpful assistant that answers questions based on the provided document excerpts.\n\nContext:\n{context}\n\nQuestion:\n{query} \n\nAnswer in a clear and concise manner using the most relevant context."""
    print("Prompt: ", prompt)
    return st.session_state.qa_chain.invoke({"question": prompt})




def main():



    st.set_page_config(page_title="Document QA",
                       page_icon="🔍")

    st.title("Document QA")
    st.write(
        "Upload multiple documents and get answers to your questions.")

    embedding_model = get_embedding_model()

    uploaded_files = st.file_uploader(
        "Choose document files (Max 5)", type="pdf", accept_multiple_files=True)

    if len(uploaded_files) > 5:
        st.warning("Please upload a maximum of 5 files at a time.")
        uploaded_files = uploaded_files[:5]

    if uploaded_files:
        try:
            with st.spinner('Processing documents...'):
                documents = process_multiple_pdfs(uploaded_files)

                if not documents:
                    st.warning("No text found in the uploaded files.")
                    return

                doc_names = list(set(doc.metadata['doc_name']
                                     for doc in documents))

                st.spinner("Loading documents...")

                st.session_state.qa_chain = create_qa_chain(documents, embedding_model)

    
            if st.session_state.qa_chain:
                question = st.chat_input("Ask a question about your document...")
                if question:
                    with st.spinner("💡 Thinking..."):
                        # result = st.session_state.qa_chain.invoke({"question": question})
                        result = rerank_results(question)
                        answer_text = result["answer"]

                        sources = result.get("source_documents", [])
                        if sources:
                            cited_docs = ", ".join(set(doc.metadata.get("doc_name", "Unknown") for doc in sources))
                            answer_text += f"\n\n**Sources cited**: {cited_docs}"

                        st.session_state.chat_history.append((question, answer_text, sources))

                for q, a, sources in st.session_state.chat_history:
                    st.chat_message("user").markdown(q)
                    st.chat_message("assistant").markdown(a)

                    if sources:
                        with st.expander("View sources"):
                            for i, doc in enumerate(sources):
                                doc_name = doc.metadata.get("doc_name", "Unknown")
                                page = doc.metadata.get("page", "N/A")
                                st.markdown(f"**Source {i+1}:** {doc_name} (Page {page})")
                                st.code(doc.page_content[:100].strip(), language="markdown")




        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload documents and enter a question to begin matching.")


if __name__ == "__main__":
    main()
