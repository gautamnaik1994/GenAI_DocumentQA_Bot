# AI-Powered Document Q&A 📚🔍

AI-Powered Document Q&A is an AI-powered document question-answering application built with Streamlit, LangChain, HuggingFace, and FAISS. It allows users to upload multiple PDF documents and interactively ask questions, receiving concise answers along with relevant source citations.

Check out the Streamlit App here [https://gen-ai-document-rag.streamlit.app/](https://gen-ai-document-rag.streamlit.app/)

Streamlit shuts down the app after some time of inactivity. Please click on "Wake Up App" button to start the app again.

## 🚀 Features

- **Multi-document Upload**: Upload up to 5 PDF documents simultaneously.
- **Conversational Q&A**: Ask questions in natural language and receive context-aware answers.
- **Source Citation**: Answers include references to the original documents and pages.
- **Relevance Ranking**: Utilizes cross-encoder models to re-rank retrieved documents for improved accuracy.
- **Memory Management**: Maintains conversational context across multiple queries.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Document Processing**: pdfplumber
- **Vector Store**: FAISS
- **Embeddings**: Sentence Transformers
- **Language Models**: LangChain, TogetherAI, LlamaCpp
- **Cross-Encoder**: sentence-transformers/ms-marco-MiniLM-L-6-v2
- **Environment Management**: python-dotenv

## 🧩 Architecture Overview

The application follows a modular architecture:

### Document Processing

- PDFs are uploaded and processed using `pdfplumber`.
- Text and tables are extracted, cleaned, and structured into LangChain Document objects.

### Embedding and Vectorization

- Document chunks are created using `RecursiveCharacterTextSplitter`.
- Embeddings are generated using HuggingFace's Sentence Transformers (`all-MiniLM-L6-v2`).
- FAISS is used as the vector store for efficient retrieval.

### Retrieval and Re-ranking

- Initial retrieval is performed using FAISS.
- A cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-ranks the retrieved documents based on relevance to the query.

### Conversational QA

- LangChain's `ConversationalRetrievalChain` integrates the retrieval system with a conversational memory buffer.
- Queries are answered using either a cloud-hosted LLM (TogetherAI's Llama-3.3-70B-Instruct-Turbo-Free) or a local LLM (Mistral-7B-Instruct).

### Memory Management

- LangChain's `ConversationBufferMemory` is used to maintain context across multiple queries.

### Frontend Interaction

- Streamlit provides an interactive UI for document upload, question input, and displaying answers with source citations.

## 🚦 Running the App Locally

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd DocumentQA
```

### 2. Set Up Environment

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root and add your API keys:

```bash
TOGETHER_AI_API_KEY=<your-api-key>
```

### 4. Run the Application

```bash
streamlit run app.py
```

Open your browser and navigate to <http://localhost:8501>.

## 📌 Usage

- Upload PDF documents using the sidebar.
- Enter your questions in the chat input box.
- View answers along with cited sources and relevance scores.

## ⚙️ Customization

- Switching LLMs: Modify `llm_handler.py` to switch between cloud and local models.
- Chunk Size & Overlap: Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` in app.py to optimize retrieval performance.

## 📖 Dependencies

Key dependencies include:

- streamlit==1.45.0
- langchain==0.3.25
- langchain-community==0.3.23
- sentence-transformers==4.1.0
- faiss-cpu==1.11.0
- pdfplumber==0.11.6
(See requirements.txt for the full list.)

## 🚧 Future Improvements

- Enhanced UI/UX with additional Streamlit components.
- Support for more document formats (e.g., DOCX, TXT).
- Integration with additional LLM providers.
