import streamlit as st 

from app.ui import pdf_uploader
from app.pdf_utils import extract_text_from_pdf, clean_text
from io import BytesIO
from app.s3_utils import process_uploaded_files, list_s3_documents, download_from_s3
#from app.config import EURI_API_KEY
from app.vectorstore_utils import create_chroma_collection, retrieve_relevant_docs, clear_chroma_collection
from app.chat_utils import get_chat_model, ask_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os 
from dotenv import load_dotenv
load_dotenv()

EURI_API_KEY=os.getenv('EURI_API_KEY')
#EURI_API_KEY = st.secrets["EURI_API_KEY"]

st.set_page_config(
    page_title="MediChat Pro - Medical Assistant",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- THEME ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet')

    html, body, [class*="st-"] {
        font-family: 'Roboto', sans-serif;
        font-size: 1.0rem;
    }

    .stApp {
        background-color: #121212;
        color: #E0E0E0;
    }

    /* Full dark sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1E1E1E !important;
        color: #f1f1f1;
        border-right: 1px solid #333333;
        padding: 1rem;
    }

    /* Keep sidebar text and elements styled */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] .stButton {
        color: #f1f1f1;
    }

    /* Removed boxed sections */
    .sidebar-box {
        background: none !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Buttons */
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #007BFF;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        width: 100%;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #0056b3;
        transform: translateY(-1px);
        box-shadow: 0 3px 10px rgba(0, 123, 255, 0.3);
    }

    /* Chat input bar */
    [data-testid="stChatInput"] {
        background-color: #1E1E1E;
        border-top: 1px solid #333333;
    }

    /* Chat bubbles */
    .st-emotion-cache-janbn0 {
        border-radius: 8px;
        padding: 10px 14px;
        box-shadow: none;
        border: none;
    }

    [data-testid="stChatMessage"]:has(div[data-testid="stAvatarIcon-user"]) {
        justify-content: end;
        display: flex;
    }

    [data-testid="stChatMessage"]:has(div[data-testid="stAvatarIcon-user"]) .st-emotion-cache-janbn0 {
        background-color: #007BFF;
        color: white;
    }

    [data-testid="stChatMessage"]:has(div[data-testid="stAvatarIcon-assistant"]) .st-emotion-cache-janbn0 {
        background-color: #2c2c2c;
        color: #F1F1F1;
    }
</style>
""", unsafe_allow_html=True)


# --- HEADER (DARK THEME) ---
st.markdown("""
<div style="text-align: center; padding: 0.5rem 0 1.5rem 0;">
    <h1 style="color: #00A6FB; font-size: 3.2rem; font-weight: 700;">🩺 MediChat </h1>
    <p style="font-size: 1.5rem; color: #A9A9A9;">Your Intelligent Medical Document Assistant</p>
</div>
""", unsafe_allow_html=True)

# Helper functions
def get_document_chunks(texts):
    """Split documents into chunks for vectorstore"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = []
    for text in texts:
        text_chunks = text_splitter.split_text(text)
        chunks.extend(text_chunks)
    return chunks

# --- APP LAYOUT ---

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
    st.header("📂 Document Uploader")
    st.markdown("📤 **Upload your PDF medical reports below:**")

    uploaded_files = pdf_uploader()

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully")
    st.markdown('</div>', unsafe_allow_html=True)

    # PROCESSING BOX
    if uploaded_files:
        st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
        if st.button("⚙️ Process Documents", type="primary"):
            with st.spinner("🔄 Processing your medical documents..."):
                # Process files: extract text and upload to S3
                process_results = process_uploaded_files(uploaded_files, extract_text_from_pdf)
                
                # Show S3 upload status
                if process_results["uploaded"]:
                    st.success(f"✅ {len(process_results['uploaded'])} file(s) uploaded to S3")
                    for item in process_results["uploaded"]:
                        st.info(f"📄 {item['filename']} → S3: {item['s3_key']}")
                
                if process_results["failed"]:
                    st.warning(f"⚠️ {len(process_results['failed'])} file(s) failed to upload")
                    for item in process_results["failed"]:
                        st.error(f"❌ {item['filename']}: {item.get('error', 'Unknown error')}")
                
                # Process texts for ChromaDB
                if process_results["texts"]:
                    documents = get_document_chunks(process_results["texts"])
                    collection = create_chroma_collection(documents)
                    st.session_state.collection = collection
                    chat_model = get_chat_model(EURI_API_KEY)
                    st.session_state.chat_model = chat_model
                    st.success("✅ Processing complete! Ready for questions.")
                else:
                    st.error("❌ No text could be extracted from the uploaded files.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show existing S3 documents
        st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
        if st.button("📋 List S3 Documents"):
            s3_docs = list_s3_documents()
            if s3_docs:
                st.write(f"**Found {len(s3_docs)} document(s) in S3:**")
                for doc in s3_docs:
                    st.write(f"📄 {doc['filename']} ({doc['size']} bytes)")
            else:
                st.info("No documents found in S3.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # S3 IMPORT SECTION
    st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
    st.header("☁️ Import from S3")
    st.markdown("📥 **Import documents from your S3 bucket:**")
    
    if st.button("🔄 Load S3 Files", type="secondary"):
        with st.spinner("🔄 Loading files from S3..."):
            s3_docs = list_s3_documents()
            if s3_docs:
                st.session_state.s3_documents = s3_docs
                st.success(f"✅ Found {len(s3_docs)} file(s) in S3")
            else:
                st.session_state.s3_documents = []
                st.warning("⚠️ No documents found in S3 bucket.")
    
    # Display S3 files and allow selection
    if "s3_documents" in st.session_state and st.session_state.s3_documents:
        st.markdown("**Select files to import:**")
        
        # Create a list of file options for selection
        file_options = [f"{doc['filename']} ({doc['size']} bytes)" for doc in st.session_state.s3_documents]
        selected_indices = st.multiselect(
            "Choose files to import:",
            options=range(len(file_options)),
            format_func=lambda x: file_options[x],
            key="s3_file_selection"
        )
        
        if selected_indices:
            if st.button("⬇️ Import Selected Files", type="primary"):
                with st.spinner("🔄 Importing and processing files from S3..."):
                    imported_texts = []
                    imported_count = 0
                    failed_count = 0
                    
                    for idx in selected_indices:
                        doc = st.session_state.s3_documents[idx]
                        s3_key = doc["key"]
                        
                        # Download file from S3
                        download_result = download_from_s3(s3_key)
                        
                        if download_result["success"]:
                            try:
                                # Extract text from downloaded PDF
                                file_bytes = download_result["content"]
                                text = extract_text_from_pdf(BytesIO(file_bytes))
                                
                                if text:
                                    imported_texts.append(text)
                                    imported_count += 1
                                    st.success(f"✅ Imported: {doc['filename']}")
                                else:
                                    st.warning(f"⚠️ No text extracted from: {doc['filename']}")
                                    failed_count += 1
                            except Exception as e:
                                st.error(f"❌ Error processing {doc['filename']}: {str(e)}")
                                failed_count += 1
                        else:
                            st.error(f"❌ Failed to download {doc['filename']}: {download_result.get('error', 'Unknown error')}")
                            failed_count += 1
                    
                    # Process imported texts for ChromaDB
                    if imported_texts:
                        documents = get_document_chunks(imported_texts)
                        collection = create_chroma_collection(documents)
                        st.session_state.collection = collection
                        chat_model = get_chat_model(EURI_API_KEY)
                        st.session_state.chat_model = chat_model
                        st.success(f"✅ Successfully imported and processed {imported_count} file(s)! Ready for questions.")
                        if failed_count > 0:
                            st.warning(f"⚠️ {failed_count} file(s) failed to import.")
                    else:
                        st.error("❌ No files could be imported successfully.")
    st.markdown('</div>', unsafe_allow_html=True)

                
                
## MAIN CHAT LOGIC 

# --- Header Section ---
# with st.container():
#     st.title("🩺Medical Health Report Chatbot")
#     st.subheader("Upload your health reports and ask questions.")
    
st.markdown("---") # Separator

# main chat interface 
st.markdown("### Chat with your Medical documents")

# 1.INITIALIZE SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages=[]
if "collection" not in st.session_state:
    st.session_state.collection=None
if "chat_model" not in st.session_state:
    st.session_state.chat_model=None
if "s3_documents" not in st.session_state:
    st.session_state.s3_documents=[]

## Sidebar Button to Clear Chat 
st.sidebar.title("Controls")
if st.sidebar.button("Clear Conversation"):
    st.session_state.messages=[]
    st.rerun() # Rerun the app to reflect the cleared messages immediately

# Initial Welcome message
if not st.session_state.messages:
    st.session_state.messages.append({
        "role":"assistant",
        "content":"Hello! I'm your medical chatbot. Please upload your PDF reports in the sidebar, and I'll help you find the information you need."
    })

## 2. DISPLAY ALL THE PAST MESSAGES
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

## 3. WAIT FOR USER INPUT AND HANDLE IT 
if prompt := st.chat_input("Ask about your medical documents..."):

    #Add user's new message to the history and display it 
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    #Check if documents have been uploaded before responding
    if st.session_state.collection:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                #RAG PIEPLINE
                context_docs=retrieve_relevant_docs(prompt, k=4)

                # Handle no relevant context ---
                if context_docs:
                    context_text="\n\n".join([doc.page_content for doc in context_docs])
                    full_prompt=f"""Based on this context:{context_text}\n\n Answer this question: {prompt}"""
                    response=ask_chat_model(st.session_state.chat_model,full_prompt)
                else:
                    response="I'm sorry, but I couldn't find any information related to your question in the uploaded documents. Could you please ask something else?"

                # Display the response
                st.markdown(response)

                ## Adding the assiant response to history
                st.session_state.messages.append({"role":"assistant","content":response})
    else:
        #Show an error if no documents are uploaded 
        st.session_state.messages.append({
            "role":"assistant",
            "content":"Please upload your documents in the sidebar first so I can answer your questions."
        })
        with st.chat_message("assistant"):
            st.error("Please upload documents in the sidebar first.")