import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pdf2image import convert_from_bytes
import fitz  # PyMuPDF for PDF processing

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:  # Ensure there is text on the page
                    text += page_text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to store text chunks in FAISS
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Conversational chain with Google Generative AI
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available, just say, "Answer is not available in the context".
    
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to highlight text in the PDF
def highlight_text_in_pdf(pdf_file, answer_text, output_path):
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")  # Open the PDF from a bytes stream
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text_instances = page.search_for(answer_text)

        if text_instances:
            # Highlight the text
            for inst in text_instances:
                highlight = page.add_highlight_annot(inst)
                highlight.update()

    # Save the highlighted PDF
    pdf_document.save(output_path)
    pdf_document.close()

# Display PDF pages
def display_pdf(pdf_file):
    try:
        pdf_bytes = pdf_file.read()  # Read the file content only once
        pdf_file.seek(0)  # Reset the file pointer to the beginning

        # Convert PDF bytes to images
        images = convert_from_bytes(pdf_bytes)
        for idx, image in enumerate(images):
            st.image(image, caption=f'Page {idx + 1}', use_column_width=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")

# User input handling and response generation
def user_input(user_question, pdf_file):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        st.write("Reply: ", response["output_text"])

        # Clear previous highlights and highlight new answer
        output_pdf = "highlighted_output.pdf"
        highlight_text_in_pdf(pdf_file, response["output_text"], output_pdf)

        # Display the updated PDF with highlights
        pdf_display = convert_from_bytes(open(output_pdf, "rb").read())
        for idx, image in enumerate(pdf_display):
            st.image(image, caption=f'Page {idx + 1} with Highlights', use_column_width=True)
    except Exception as e:
        st.error(f"Error processing user input: {str(e)}")

# Main app functionality
def main():
    st.set_page_config("Chat PDF & Webpages")
    st.header("Chat with PDF and Web Links using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files or Webpages")
    pdf_docs = st.sidebar.file_uploader("Upload your PDF Files", accept_multiple_files=False, type=["pdf"])

    if st.sidebar.button("Submit & Process") and pdf_docs:
        with st.spinner("Processing..."):
            # Process PDF files
            raw_text = ""
            if pdf_docs:
                # Use the uploaded PDF file
                raw_text += get_pdf_text([pdf_docs])  # Extract text from the PDF

            if raw_text:
                # Split and store text
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing complete! You can now ask questions.")
                
                # Display PDF pages on the frontend
                display_pdf(pdf_docs)
            else:
                st.warning("No text extracted from the PDF. Please check the file.")

    if user_question and pdf_docs:
        # Ask a question and get the highlighted PDF
        user_input(user_question, pdf_docs)

if __name__ == "__main__":
    main()
