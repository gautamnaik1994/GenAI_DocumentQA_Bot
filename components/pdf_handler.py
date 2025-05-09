
import pdfplumber
import tempfile
from langchain.docstore.document import Document
from typing import List
import os
import re


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
                            table_str = '\n'.join(
                                [' | '.join(row) for row in table if any(row)])
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
