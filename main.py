import streamlit as st 

from app.ui import pdf_uploader
from app.chat_utils import get_chat_model,ask_chat_model
#from app.config import EURI_API_KEY
from app.pdf_utils import load_documents_from_pdfs,get_document_chunks
from app.vectorstore_utils import create_faiss_index,retrieve_relevant_docs
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
                all_documents = load_documents_from_pdfs(uploaded_files)
                documents = get_document_chunks(all_documents)
                vectorstore = create_faiss_index(documents)
                st.session_state.vectorstore = vectorstore
                chat_model = get_chat_model(EURI_API_KEY)
                st.session_state.chat_model = chat_model
                st.success("✅ Processing complete! Ready for questions.")
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
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore=None
if "chat_model" not in st.session_state:
    st.session_state.chat_model=None

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
    if st.session_state.vectorstore:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                #RAG PIEPLINE
                context_docs=retrieve_relevant_docs(st.session_state.vectorstore, prompt)

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