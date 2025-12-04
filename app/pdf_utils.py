import os 
import tempfile 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents_from_pdfs(pdf_files):
    """
    Loads one or more uploaded PDF files and extracts their content into LangChain Document objects.

    Args:
        pdf_files: A list of uploaded file objects from Streamlit's file_uploader.

    Returns:
        list: A list of LangChain Document objects, where each object represents a page
              from all the uploaded PDFs. Returns an empty list if processing fails.
    """
    all_documents=[]

    for pdf_file in pdf_files:
        # PyPDFLoader requires a file path. We save each uploaded file to a
        # temporary file on disk to get a path.
        temp_file_path=""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(pdf_file.getvalue())
                temp_file_path=temp_file.name

            #Use a PDFLoadr to load the temporary files 
            loader=PyPDFLoader(temp_file_path)
            documents=loader.load()
            all_documents.extend(documents)# Add the documents from the current PDF to the list
        except Exception as e:
            print(f"Error processing file {pdf_file.name} :{e}")

        finally:
            #Clean up the temporary file after processing 
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    return all_documents

## We can also write a functon to split the document objects into chunks using Langchain
def get_document_chunks(documents):
    if not documents:
        return []
    
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=30,
        length_function=len,
    )

    chunks=text_splitter.split_documents(documents)
    return chunks 
