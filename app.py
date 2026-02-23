import streamlit as st
import sqlite3
import os
import tempfile
from dotenv import load_dotenv


# --- Naye Modern Imports ---
from qdrant_client.http import models
from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

# --- 1. Config & Setup ---
load_dotenv()

# Streamlit Page Config (Wide for Dashboard)
st.set_page_config(page_title="Havi - Enterprise AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# Semantic Caching
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

HF_TOKEN = os.getenv("HF_TOKEN")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=HF_TOKEN
)

# ------------------------------------------------
# 2. Database Setup & CSS UI Theme
# ------------------------------------------------
# --- ADMIN & DATABASE FUNCTIONS (Paste here) ---
def init_db():
    conn = sqlite3.connect('enterprise_rag.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT UNIQUE, password TEXT, role TEXT)''')
    
    # Check if admin exists
    cursor.execute("SELECT * FROM users WHERE username='admin'")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO users VALUES ('admin', 'admin123', 'Admin')") # SUPER USER
        cursor.execute("INSERT OR IGNORE INTO users VALUES ('ali', 'pass123', 'HR')")
        cursor.execute("INSERT OR IGNORE INTO users VALUES ('sara', 'pass123', 'Finance')")
        cursor.execute("INSERT OR IGNORE INTO users VALUES ('ahmed', 'pass123', 'Support')")
    conn.commit()
    conn.close()

def verify_login(username, password):
    conn = sqlite3.connect('enterprise_rag.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = cursor.fetchone()
    conn.close()
    return user

def get_all_users():
    conn = sqlite3.connect('enterprise_rag.db')
    cursor = conn.cursor()
    cursor.execute("SELECT username, role FROM users WHERE role != 'Admin'")
    users = cursor.fetchall()
    conn.close()
    return users

def add_new_user(username, password, role):
    conn = sqlite3.connect('enterprise_rag.db')
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users VALUES (?, ?, ?)", (username, password, role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def delete_existing_user(username):
    conn = sqlite3.connect('enterprise_rag.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE username = ?", (username,))
    conn.commit()
    conn.close()

def delete_qdrant_department_data(department):
    try:
        q_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60.0)
        q_client.delete(
            collection_name="enterprise_knowledge",
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(key="metadata.department", match=models.MatchValue(value=department))])
            )
        )
        return True
    except Exception as e:
        return str(e)

init_db() # Database initialize karna zaroori hai
# Custom CSS matching the Image EXACTLY
st.markdown("""
<style>
    /* Dark Theme Reset */
    [data-testid="stAppViewContainer"] { background-color: #0E1117; color: #E0E0E0; }
    [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #2D333B; }
    [data-testid="stHeader"] { background-color: transparent; }
    #MainMenu, footer, header { visibility: hidden; }
    
    /* Typography */
    h1, h2, h3, p { color: #E0E0E0; font-family: 'Inter', sans-serif; }
    
    /* Sidebar Styling */
    .sidebar-branding { display: flex; align-items: center; margin-bottom: 2rem; }
    .branding-icon { font-size: 1.8rem; margin-right: 0.8rem; }
    .branding-text { font-size: 1.3rem; font-weight: 600; color: #fff; }
    
    /* Sidebar Buttons */
    [data-testid="stSidebar"] .stButton > button {
        width: 100%; border-radius: 8px; font-weight: 500; border: none; padding: 0.6rem 1rem;
    }
    .btn-primary > button { background-color: #2F81F7 !important; color: white !important; }
    .btn-secondary > button { background-color: transparent !important; border: 1px solid #30363D !important; color: #8B949E !important; }
    .btn-secondary > button:hover { border-color: #8B949E !important; color: #E0E0E0 !important; }
    
    /* Top Search Bar & Alerts */
    .top-search-bar { background-color: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 0.6rem 1rem; color: #8B949E; }
    .alert-banner { padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; font-size: 0.95rem; }
    .alert-blue { background-color: rgba(47, 129, 247, 0.1); border: 1px solid #2F81F7; color: #58A6FF; }
    .alert-red { background-color: rgba(248, 81, 73, 0.1); border: 1px solid #F85149; color: #FF7B72; }
    
    /* Feature Cards */
    .feature-card { background-color: #161B22; border: 1px solid #30363D; border-radius: 12px; padding: 1.5rem; transition: all 0.3s; height: 100%; }
    .feature-card:hover { border-color: #58A6FF; transform: translateY(-2px); }
    .card-icon { font-size: 1.2rem; background: rgba(88, 166, 255, 0.1); padding: 0.5rem; border-radius: 8px; display: inline-block; margin-bottom: 1rem; color: #58A6FF; }
    .card-title { font-weight: 600; font-size: 1.05rem; margin-bottom: 0.5rem; color: #E0E0E0; }
    .card-desc { font-size: 0.85rem; color: #8B949E; line-height: 1.4; }
    
    /* Chat Input Styling */
    .stChatInput > div { background-color: #161B22; border: 1px solid #30363D; border-radius: 12px; box-shadow: 0 0 15px rgba(88,166,255,0.05); }
</style>
""", unsafe_allow_html=True)

# Session State Init
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "username" not in st.session_state: st.session_state.username = ""
if "role" not in st.session_state: st.session_state.role = ""
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# ------------------------------------------------
# 3. Application Flow
# ------------------------------------------------

# --- LOGIN SCREEN ---
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.write("<br><br><br>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>🧠 Havi</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #8B949E;'>Secure Enterprise Knowledge Assistant</p>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Department ID", placeholder="Enter username (e.g., ahmed)")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Access Workspace", use_container_width=True)
            
            if submitted:
                if username and password:
                    user_data = verify_login(username, password)
                    if user_data:
                        st.session_state.logged_in = True
                        st.session_state.username = user_data[0]
                        st.session_state.role = user_data[2]
                        st.rerun()
                    else:
                        st.error("Invalid Department ID or Password.")
                else:
                    st.warning("Fill in all fields.")

# --- MAIN DASHBOARD (Matching Image) ---
else:
    # --- SIDEBAR ---
    with st.sidebar:
        # Branding
        st.markdown("""
            <div class="sidebar-branding">
                <span class="branding-icon">🧠</span>
                <span class="branding-text">Havi</span>
            </div>
            <p style='color: #8B949E; font-size: 0.9rem; margin-top:-1.5rem;'><span style='margin-right:8px;'>⚙️</span> Havi Admin Panel</p>
            <hr style="border-color:#30363D;">
        """, unsafe_allow_html=True)
        
        # Chat Controls
        st.markdown("<p style='font-size: 0.9rem; color: #E0E0E0; font-weight:500;'>💬 Chat Controls</p>", unsafe_allow_html=True)
        st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        st.markdown('</div><div class="btn-secondary">', unsafe_allow_html=True)
        if st.button("Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<hr style='border-color:#30363D;'>", unsafe_allow_html=True)
        
        # Document Management (Actual Upload Logic)
        st.markdown("<p style='font-size: 0.9rem; color: #E0E0E0; font-weight:500;'>📤 Document Management</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 0.8rem; color: #8B949E; margin-bottom: 0px;'>Upload Document</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drag & drop", type=["pdf", "csv", "txt"], label_visibility="collapsed")
        
        st.markdown("<p style='font-size: 0.8rem; color: #8B949E; margin-bottom: 0px; margin-top: 10px;'>Select Department Tag</p>", unsafe_allow_html=True)
        target_role = st.selectbox("Tag", ["HR", "Finance", "Support"], label_visibility="collapsed")
        
        if st.button("Upload to Cloud Database", use_container_width=True):
            if uploaded_file is not None:
                with st.spinner("Processing & Uploading..."):
                    # Original Upload Logic
                    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    if file_extension == '.pdf': loader =PyMuPDFLoader(tmp_path)
                    elif file_extension == '.csv': loader = CSVLoader(tmp_path, encoding="utf-8")
                    else: loader = TextLoader(tmp_path)
                    
                    docs = loader.load()
                    docs = docs[:30] # Testing limit
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = text_splitter.split_documents(docs)
                    for chunk in chunks: chunk.metadata["department"] = target_role

                    try:
                        q_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120.0)
                        try:
                            q_client.create_payload_index(collection_name="enterprise_knowledge", field_name="metadata.department", field_schema=models.PayloadSchemaType.KEYWORD)
                        except: pass

                        batch_size = 5 
                        for i in range(0, len(chunks), batch_size):
                            batch = chunks[i : i + batch_size]
                            QdrantVectorStore.from_documents(batch, embeddings, url=QDRANT_URL, api_key=QDRANT_API_KEY, collection_name="enterprise_knowledge", timeout=120.0)
                        st.success(f"{len(chunks)} chunks uploaded!")
                    except Exception as upload_error:
                        st.error(f"Upload fail: {upload_error}")
                    os.remove(tmp_path)
            else:
                st.warning("Upload a file first.")
        
        # Profile Footer
        st.markdown("<div style='height: 10vh;'></div>", unsafe_allow_html=True) # Spacer
        st.markdown(f"""
        <div style="display: flex; align-items: center; background-color: #161B22; padding: 0.8rem; border-radius: 8px; border: 1px solid #30363D; margin-bottom: 10px;">
            <div style="background-color: #2F81F7; color: white; width: 32px; height: 32px; border-radius: 50%; display: flex; justify-content: center; align-items: center; margin-right: 10px; font-weight: bold;">
                {st.session_state.username[0].upper()}
            </div>
            <div>
                <div style="color: #E0E0E0; font-weight: 600; font-size: 0.9rem;">{st.session_state.username.capitalize()}</div>
                <div style="color: #8B949E; font-size: 0.75rem;">Welcome, {st.session_state.username.capitalize()}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="btn-secondary">', unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.chat_history = []
            st.session_state.username = ""
            st.session_state.role = ""
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- MAIN WORKSPACE AREA ---
    # Header Top Bar
    c1, c2, c3 = st.columns([1, 2, 1])
    with c3: st.markdown(f"<div style='text-align: right; color:#E0E0E0; margin-top:0.5rem;'>👤 Welcome, {st.session_state.username.capitalize()}</div>", unsafe_allow_html=True)

    # Main Banners
    # ==========================================
    # --- ADMIN EXCLUSIVE DASHBOARD ---
    # ==========================================
    if st.session_state.role == "Admin":
        import pandas as pd # Dataframe dikhane ke liye
        
        st.markdown("<h2>🛡️ Global Admin Dashboard</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#8B949E; margin-bottom:2rem;'>Manage user roles and control vector database integrity.</p>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["👥 User Management", "🗑️ Vector Database Control"])
        
        with tab1:
            col_u1, col_u2 = st.columns([2, 1])
            with col_u1:
                st.markdown("#### Current Active Users")
                users = get_all_users()
                df = pd.DataFrame(users, columns=["Username", "Department Role"])
                st.dataframe(df, use_container_width=True, hide_index=True)
            
            with col_u2:
                st.markdown("#### Add New User")
                with st.form("add_user_form", clear_on_submit=True):
                    new_user = st.text_input("New Username")
                    new_pass = st.text_input("Password", type="password")
                    new_role = st.selectbox("Assign Role", ["HR", "Finance", "Support"])
                    if st.form_submit_button("Create User"):
                        if new_user and new_pass:
                            if add_new_user(new_user, new_pass, new_role):
                                st.success(f"User '{new_user}' created! Refreshing...")
                                st.rerun()
                            else:
                                st.error("Username already exists!")
                        else:
                            st.warning("Fill all fields.")
                            
                st.markdown("#### Revoke Access")
                with st.form("delete_user_form", clear_on_submit=True):
                    del_user = st.text_input("Username to delete")
                    if st.form_submit_button("Delete User"):
                        if del_user:
                            delete_existing_user(del_user)
                            st.success(f"User '{del_user}' deleted. Refreshing...")
                            st.rerun()

        with tab2:
            st.markdown("#### Emergency Data Erasure")
            st.warning("⚠️ **Warning:** This action will permanently delete all vector embeddings associated with a specific department from the Qdrant Cloud. This cannot be undone.")
            
            with st.form("delete_data_form"):
                dept_to_delete = st.selectbox("Select Department Data to Purge", ["HR", "Finance", "Support"])
                confirm_check = st.checkbox(f"I confirm I want to delete ALL data tagged as {dept_to_delete}")
                
                if st.form_submit_button("Purge Vectors"):
                    if confirm_check:
                        with st.spinner(f"Deleting all {dept_to_delete} vectors from Qdrant..."):
                            result = delete_qdrant_department_data(dept_to_delete)
                            if result is True:
                                st.success(f"✅ Successfully deleted all vector records for {dept_to_delete}.")
                            else:
                                st.error(f"Failed to delete: {result}")
                    else:
                        st.error("You must check the confirmation box to proceed.")

    # ==========================================
    # --- REGULAR USER WORKSPACE ---
    # ==========================================
    elif st.session_state.role != "Admin":
        st.markdown("""
            <div style="background-color: #161B22; border: 1px solid #30363D; border-radius: 12px; padding: 2rem; margin-bottom: 2rem;">
                <h2 style="margin-top:0;">🧠 Havi Workspace</h2>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
                <div class="alert-banner alert-blue">Security Level Active: You are securely restricted to <b>{st.session_state.role}</b> department data only.</div>
                <div class="alert-banner alert-red">Your Department Role: <b>{st.session_state.role}</b></div>
            </div>
        """, unsafe_allow_html=True)

        # If Chat History is empty, show the 4 Cards, otherwise hide them to focus on chat
        if len(st.session_state.chat_history) == 0:
            st.markdown(f"<div style='background-color:#161B22; border:1px solid #30363D; border-radius:8px; padding:0.8rem 1rem; color:#8B949E; margin-bottom:2rem;'>Ask a query about {st.session_state.role} department documents... 🔍</div>", unsafe_allow_html=True)
            
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown(f"""
                <div class="feature-card"><div class="card-icon">📄</div><div class="card-title">{st.session_state.role} Docs</div><div class="card-desc">Create documents and ask questions to {st.session_state.role} docs.</div></div>
                """, unsafe_allow_html=True)
            with cc2:
                st.markdown("""
                <div class="feature-card"><div class="card-icon">📖</div><div class="card-title">Knowledge Base</div><div class="card-desc">Learn more about knowledge base architectures.</div></div>
                """, unsafe_allow_html=True)
            
            st.write("")
            cc3, cc4 = st.columns(2)
            with cc3:
                st.markdown("""
                <div class="feature-card"><div class="card-icon">👥</div><div class="card-title">User Management</div><div class="card-desc">User management and user flow controls.</div></div>
                """, unsafe_allow_html=True)
            with cc4:
                st.markdown("""
                <div class="feature-card"><div class="card-icon">📊</div><div class="card-title">Analytics</div><div class="card-desc">Analyze configuration and analytics.</div></div>
                """, unsafe_allow_html=True)

        # Chat History Rendering
        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]): st.write(msg["content"])

        # Chat Input & RAG Logic (Floating at bottom)
        if user_q := st.chat_input(f"Ask a question related to {st.session_state.role}..."):
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.chat_message("user"): st.write(user_q)
                
            with st.chat_message("assistant"):
                with st.spinner("Searching secure database..."):
                    try:
                        # Original RAG Logic
                        llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
                        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60.0)
                        qdrant_store = QdrantVectorStore(client=client, collection_name="enterprise_knowledge", embedding=embeddings)
                        
                        rbac_filter = models.Filter(must=[models.FieldCondition(key="metadata.department", match=models.MatchValue(value=st.session_state.role))])
                        retriever = qdrant_store.as_retriever(search_kwargs={"filter": rbac_filter})
                        
                        prompt = ChatPromptTemplate.from_template("""
                        You are a helpful enterprise assistant. Answer the question based ONLY on the provided context.
                        If the answer is not in the context, strictly say "I don't have authorization or information to answer this based on your department's records."
                        
                        <context>\n{context}\n</context>
                        Question: {input}
                        """)
                        
                        def format_docs(docs): return "\n\n".join(doc.page_content for doc in docs)
                        rag_chain = ({"context": retriever | format_docs, "input": RunnablePassthrough()} | prompt | llm | StrOutputParser())
                        
                        response = rag_chain.invoke(user_q)
                        st.write(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        st.error(f"System Error: {e}")