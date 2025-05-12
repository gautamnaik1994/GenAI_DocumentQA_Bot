import os
import re
import streamlit as st
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS

from sentence_transformers import CrossEncoder

from components.pdf_handler import process_multiple_pdfs
from components.llm_handler import get_llm
from components.memory_handler import get_memory

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 100


if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


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
    text = re.sub(r'\n(?=(Introduction|Abstract|Conclusion|References|Appendix))',
                  r'\n\n', text, flags=re.IGNORECASE)
    return text


def create_qa_chain(documents, embedding_model):
    text_splitter = get_text_splitter()

    for doc in documents:
        doc.page_content = normalize_section_headers(doc.page_content)

    splits = text_splitter.split_documents(documents)

    if not splits:
        return None

    vector_store = FAISS.from_documents(
        documents=splits,
        embedding=embedding_model
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(use_cloud=True),
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        memory=get_memory(),
        return_source_documents=True,
        output_key="answer",
    )

    return qa_chain


def format_chat_history(chat_history: list) -> str:
    return "\n".join([f"User: {q}\nAssistant: {r}" for q, r, a, _ in chat_history])


def re_rank_results(query, cross_encoder):
    docs = st.session_state.qa_chain.retriever.invoke(query)

    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    ranked_docs_with_scores = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )
    top_docs_with_scores = ranked_docs_with_scores[:3]

    context = "\n\n".join(
        [doc.page_content for _, doc in top_docs_with_scores])
    prompt = f"""You are a helpful assistant that answers questions based on the provided document excerpts and chat history.\n\n
    Chat History:\n{format_chat_history(st.session_state.chat_history)}\n\n
    Context:\n{context}\n\nQuestion:\n{query} \n\n
    Answer in a clear and concise manner using the most relevant context and chat history. Use the chat history for extracting entities and aspects.
    """
    print(prompt)
    result = st.session_state.qa_chain.invoke({"question": prompt})

    return result, top_docs_with_scores


def main():
    st.set_page_config(page_title="AI-Powered Document Q&A",
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
    col1, col2 = st.columns([6, 1], vertical_alignment="bottom")
    with col1:
        st.title("AI-Powered Document Q&A")
    with col2:

        if st.button("New Chat", use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.get("qa_chain"):
                st.session_state.qa_chain.memory.clear()
        # st.rerun()

    embedding_model = get_embedding_model()
    cross_encoder = get_cross_encoder()

    with st.sidebar:
        st.subheader("Upload Documents")

        uploaded_files = st.file_uploader(
            "Choose document files (Max 5)", type="pdf", accept_multiple_files=True)

        if len(uploaded_files) > 5:
            st.warning("Please upload a maximum of 5 files at a time.")
            uploaded_files = uploaded_files[:5]

        if uploaded_files and "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = uploaded_files
            st.session_state.processed_documents = None

    if uploaded_files:
        try:
            if st.session_state.processed_documents is None:
                with st.spinner('Processing documents...'):
                    documents = process_multiple_pdfs(uploaded_files)

                    if not documents:
                        st.warning("No text found in the uploaded files.")
                        return

                    st.session_state.processed_documents = documents

            documents = st.session_state.processed_documents

            if "qa_chain" not in st.session_state or st.session_state.qa_chain is None:
                st.session_state.qa_chain = create_qa_chain(
                    documents, embedding_model)

            if st.session_state.qa_chain:
                question = st.chat_input(
                    "Ask a question about your document...")
                if question:
                    with st.spinner("Thinking..."):
                        # result = st.session_state.qa_chain.invoke({"question": question})
                        result, top_docs_with_scores = re_rank_results(
                            question, cross_encoder)
                        raw_answer = result["answer"]
                        answer_text = raw_answer
                        sources = result.get("source_documents", [])
                        if sources:
                            cited_docs = ", ".join(set(doc.metadata.get(
                                "doc_name", "Unknown") for doc in sources))
                            answer_text += f"\n\n**Sources cited**: {cited_docs}"

                        st.session_state.chat_history.append(
                            (question, raw_answer, answer_text, sources))

                for q, r, answer_text, sources in st.session_state.chat_history:
                    st.chat_message("user").markdown(q)
                    st.chat_message("assistant").markdown(answer_text)

                    if top_docs_with_scores:
                        with st.expander("View sources and relevance scores"):
                            for i, (score, doc) in enumerate(top_docs_with_scores):
                                doc_name = doc.metadata.get(
                                    "doc_name", "Unknown")
                                page = doc.metadata.get("page", "N/A")
                                st.markdown(
                                    f"**Source {i+1}:** {doc_name} (Page {page}) — Score: `{score:.2f}`")
                                st.code(
                                    doc.page_content[:100].strip(), language="markdown")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload documents in the sidebar to get started")


if __name__ == "__main__":
    main()
