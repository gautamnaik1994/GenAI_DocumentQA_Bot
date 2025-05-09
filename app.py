import os
import re
import tempfile
from typing import List
import streamlit as st
from dotenv import load_dotenv

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
# from langchain_community.llms import LlamaCpp
from sentence_transformers import CrossEncoder

from components.pdf_handler import process_multiple_pdfs

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
# MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
LOCAL_MODEL_PATH = "/Users/gautamnaik/models/Mistral-7B-Instruct-v0.3.Q8_0.gguf"



if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []




def get_cloud_llm():
    return ChatTogether(
        api_key=os.getenv("TOGETHER_AI_API_KEY"),
        temperature=0.0,
        model=MODEL_NAME
    )


# def get_local_llm():
#     llm = LlamaCpp(
#         model_path=LOCAL_MODEL_PATH,
#         n_ctx=2048,
#         n_threads=6,
#         n_gpu_layers=32,
#         temperature=0.7
#     )
#     return llm


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

        
    vector_store = FAISS.from_documents(
        documents=splits,
        embedding=embedding_model
    )

    memory = get_memory()

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=get_cloud_llm(),
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True,
        output_key="answer",
    )

    return qa_chain

def re_rank_results(query, cross_encoder):
    docs = st.session_state.qa_chain.retriever.get_relevant_documents(query)

    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    ranked_docs_with_scores = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )
    top_docs_with_scores = ranked_docs_with_scores[:3]

    context = "\n\n".join([doc.page_content for _, doc in top_docs_with_scores])
    prompt = f"""You are a helpful assistant that answers questions based on the provided document excerpts.\n\nContext:\n{context}\n\nQuestion:\n{query} \n\nAnswer in a clear and concise manner using the most relevant context."""

    result = st.session_state.qa_chain.invoke({"question": prompt})

    return result, top_docs_with_scores



def main():

    st.set_page_config(page_title="Document QA",
                       page_icon="🔍", layout="wide")
    st.markdown(
    """
    <style>
        .stSidebar {
            width: 500px!important;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([6,1], vertical_alignment="bottom")
    with col1:
        st.title("Document Q&A")
    with col2:
  
        if st.button("New Chat", use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.get("qa_chain"):
                st.session_state.qa_chain.memory.clear()
        # st.rerun()
 

    embedding_model = get_embedding_model()
    cross_encoder = get_cross_encoder()

    with st.sidebar:
    
        st.write(
        "Upload multiple documents and get answers to your questions.")
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

                st.session_state.qa_chain = create_qa_chain(documents, embedding_model)

    
            if st.session_state.qa_chain:
                question = st.chat_input("Ask a question about your document...")
                if question:
                    with st.spinner("Thinking..."):
                        # result = st.session_state.qa_chain.invoke({"question": question})
                        result, top_docs_with_scores = re_rank_results(question, cross_encoder)
                        answer_text = result["answer"]

                        sources = result.get("source_documents", [])
                        if sources:
                            cited_docs = ", ".join(set(doc.metadata.get("doc_name", "Unknown") for doc in sources))
                            answer_text += f"\n\n**Sources cited**: {cited_docs}"

                        st.session_state.chat_history.append((question, answer_text, sources))

                for q, a, sources in st.session_state.chat_history:
                    st.chat_message("user").markdown(q)
                    st.chat_message("assistant").markdown(a)

                    if top_docs_with_scores:
                        with st.expander("View sources and relevance scores"):
                            for i, (score, doc) in enumerate(top_docs_with_scores):
                                doc_name = doc.metadata.get("doc_name", "Unknown")
                                page = doc.metadata.get("page", "N/A")
                                st.markdown(f"**Source {i+1}:** {doc_name} (Page {page}) — Score: `{score:.2f}`")
                                st.code(doc.page_content[:100].strip(), language="markdown")
                

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload documents in the sidebar to get started")


if __name__ == "__main__":
    main()
