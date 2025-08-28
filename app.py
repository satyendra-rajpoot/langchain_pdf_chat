from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain import hub

from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the provided context to answer the question."),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask PDF")
    st.header("Ask PDF with RAG")

    # Upload PDF
    pdf = st.file_uploader("Upload a PDF file.", type="pdf")

    if pdf is not None:
        # Extract text
        pdf_reader = PdfReader(pdf)
        text = "".join([page.extract_text() or "" for page in pdf_reader.pages])

        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Get user question
        user_question = st.text_input("Ask a question from the uploaded PDF:")

        if user_question:
            # Retrieve relevant docs
            docs = knowledge_base.similarity_search(user_question, k=4)

            # Pull RAG prompt template
            prompt = hub.pull("rlm/rag-prompt")

            # Format docs as context
            context = "\n\n".join([d.page_content for d in docs])

            # Prepare messages
            messages = RAG_PROMPT.invoke({"question": user_question, "context": context})

            # LLM (use gpt-4o-mini or gpt-3.5-turbo)
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

            # Get response
            response = llm.invoke(messages)

            st.subheader("Answer:")
            st.write(response.content)


if __name__ == "__main__":
    main()
