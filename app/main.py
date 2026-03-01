"""
Main Streamlit Application
PDF RAG system using Endee vector database and Gemini AI.
Each PDF gets its own database index.
"""

import streamlit as st
import os
import re
import time
from docx import Document

from endee_client import EndeeClient
from rag import RAGSystem
from config import API_KEY, ENDEE_BASE_URL, CACHE_DIR

# ---- Helper Functions ----
def sanitize_filename(filename):    
    name = os.path.splitext(filename)[0]
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    name = name.lower()[:50]
    return f"rag_{name}"

def extract_text_from_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8", errors="ignore")

def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    return "\n".join([p.text for p in doc.paragraphs])


# ---- Initialize Systems ----
rag_system = RAGSystem(api_key=API_KEY)
endee_client = EndeeClient(base_url=ENDEE_BASE_URL)


# ---- Database Health Check ----
db_active, db_error = endee_client.is_database_active()

if db_active:
    pass
else:
    st.sidebar.error("🔴 Database Offline")
    st.sidebar.caption("Start Endee at http://localhost:8080")
    

# ---- UI Setup ----
st.set_page_config(page_title="RAG", layout="wide")

# st.title("📄 RAG System with Endee vector database")
st.title("📄 Endee Vector RAG Engine")
st.sidebar.title("System Monitoring")

# Initializing session state
if "ready" not in st.session_state:
    st.session_state.ready = False
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None
if "current_index" not in st.session_state:
    st.session_state.current_index = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- File Upload ----
uploaded = st.file_uploader("Upload Document", type=["pdf", "txt", "docx"])

if uploaded:
    # Generate unique index name for this PDF
    pdf_filename = uploaded.name
    index_name = sanitize_filename(pdf_filename)
    cache_file = f"{index_name}.pkl"
    
    # Check if this is a new PDF - reset everything if it is
    if st.session_state.current_pdf != pdf_filename:
        st.session_state.ready = False
        st.session_state.current_pdf = pdf_filename
        st.session_state.current_index = index_name
        st.session_state.chat_history = []  # Reset chat history
        
        # Update endee client with new index name
        endee_client.set_index_name(index_name)

        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(CACHE_DIR, f"{index_name}.pkl")
        endee_client.set_cache_file(cache_file)
        # endee_client.set_cache_file(cache_file)
    
    # Show current PDF info
    st.info(f"**Current Document:** {pdf_filename}")
    
    # Check if this PDF is already indexed
    if endee_client.index_exists() and os.path.exists(os.path.join(CACHE_DIR, cache_file)):
        st.session_state.ready = True
        st.success("This Document is already indexed and ready for questions.")
        
        # Show database info in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### DATABASE INFORMATION")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.markdown("<small>**PDF Name**</small>", unsafe_allow_html=True)
            st.markdown("<small>**Index ID**</small>", unsafe_allow_html=True)
            st.markdown("<small>**Vector Count**</small>", unsafe_allow_html=True)
            st.markdown("<small>**Status**</small>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<small>{pdf_filename[:15]}...</small>", unsafe_allow_html=True)
            st.markdown(f"<small>`{index_name[:15]}...`</small>", unsafe_allow_html=True)
            st.markdown(f"<small>{endee_client.get_vector_count():,}</small>", unsafe_allow_html=True)
            st.markdown("<small style='color: green;'>Active</small>", unsafe_allow_html=True)
        
        # Option to re-index
        if st.sidebar.button("Re-index PDF", use_container_width=True):
            st.session_state.ready = False
            st.session_state.chat_history = []
            st.rerun()
    
    # Index the PDF if not ready
    if not st.session_state.ready:
        step_times = {}
        total_start = time.perf_counter()
        # Create sidebar status area
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### INDEXING PROGRESS")
        status_container = st.sidebar.container()
        
        with st.spinner("Processing ..."):
            try:
                # Step 1: Extract text from PDF
                with status_container:
                    st.markdown("<small>**[1/5]** Extracting text from document...</small>", unsafe_allow_html=True)
                start = time.perf_counter()
                # chunks = rag_system.process_pdf(uploaded)

                file_ext = os.path.splitext(uploaded.name)[1].lower()

                if file_ext == ".pdf":
                    chunks = rag_system.process_pdf(uploaded)

                elif file_ext == ".txt":
                    text = extract_text_from_txt(uploaded)
                    chunks = rag_system.chunk_text(text)

                elif file_ext == ".docx":
                    text = extract_text_from_docx(uploaded)
                    chunks = rag_system.chunk_text(text)

                else:
                    st.error("Unsupported file type")
                    st.stop()

                step_times["Text Extraction"] = time.perf_counter() - start
                
                
                # Step 2: Generate embeddings
                with status_container:
                    st.markdown("<small>**[2/5]** Generating embeddings...</small>", unsafe_allow_html=True)
                start = time.perf_counter()
                vectors = rag_system.embed_texts(chunks)
                step_times["Embedding Generation"] = time.perf_counter() - start
                # vectors = rag_system.embed_texts(chunks)
                
                # Step 3: Check/Create index
                with status_container:
                    st.markdown("<small>**[3/5]** Setting up vector database...</small>", unsafe_allow_html=True)
                start = time.perf_counter()
                if not endee_client.index_exists():
                    endee_client.create_index(dimension=len(vectors[0]))
                step_times["Index Setup"] = time.perf_counter() - start
            
                
                # Step 4: Upload vectors
                with status_container:
                    st.markdown("<small>**[4/5]** Storing vectors in database...</small>", unsafe_allow_html=True)
                start = time.perf_counter()
                num_uploaded = endee_client.upsert_vectors(chunks, vectors)
                step_times["Vector Upload"] = time.perf_counter() - start
                # num_uploaded = endee_client.upsert_vectors(chunks, vectors)
                
                # Step 5: Finalize
                with status_container:
                    st.markdown("<small>**[5/5]** Finalizing index...</small>", unsafe_allow_html=True)
                step_times["Finalization"] = time.perf_counter() - total_start
                # Clear status and show final info
                status_container.empty()
                st.sidebar.markdown("#### DATABASE INFORMATION")
                
                # Create metrics table
                st.sidebar.markdown(f"""
                <small>
                
                | Metric | Value |
                |--------|-------|
                | Document Name | {pdf_filename[:20]}... |
                | Index ID | `{index_name[:20]}...` |
                | Text Chunks | {len(chunks):,} |
                | Vectors Stored | {num_uploaded:,} |
                | Vector Dimension | {len(vectors[0])} |
                | Embedding Model | gemini-embedding-001 |
                | Status | <span style='color: green;'>Ready</span> |
                
                </small>
                """, unsafe_allow_html=True)
                
                st.sidebar.markdown("---")
                st.sidebar.markdown("#### ⏱️ INDEXING PERFORMANCE")

                st.sidebar.markdown("<small>", unsafe_allow_html=True)

                for step, duration in step_times.items():
                    st.sidebar.markdown(
                        f"<small>• {step}: `{duration:.2f}s`</small>",
                        unsafe_allow_html=True
                    )

                total_time = sum(step_times.values())

                st.sidebar.markdown(
                    f"<small><b>Total Time:</b> `{total_time:.2f}s`</small>",
                    unsafe_allow_html=True
                )

                st.sidebar.markdown("</small>", unsafe_allow_html=True)


                # Mark as ready
                st.session_state.ready = True
                st.success(f"Successfully indexed {len(chunks):,} chunks from the document!")
                
            except ValueError as e:
                st.sidebar.error(f"Error: {str(e)}")
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.sidebar.error(f"Processing failed: {str(e)}")
                st.error(f"Error processing document: {str(e)}")
                st.stop()

# ---- Question Input ----
# Only show chat interface if PDF is ready
if st.session_state.ready:
    st.markdown("---")
    st.markdown("### Chat with your document")
    
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
    
    # Chat input
    question = st.chat_input("Ask a question about the document...")
    
    if question:
        query_times = {}
        query_start_total = time.perf_counter()
        # Ensure the index is set correctly
        if st.session_state.current_index:
            endee_client.set_index_name(st.session_state.current_index)
            cache_file = f"{st.session_state.current_index}.pkl"
            endee_client.set_cache_file(cache_file)
        
        # Show user message immediately
        with st.chat_message("user"):
            st.write(question)
        
        # Create query status in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### QUERY PROCESSING")
        query_status = st.sidebar.empty()

        proceed = True
        with st.spinner("Processing..."):
            try:
                # Step 1: Generate query embedding
                query_status.markdown("<small>**[1/4]** Embedding query...</small>", unsafe_allow_html=True)
                start = time.perf_counter()
                query_vector = rag_system.embed_single(question)
                query_times["Query Embedding"] = time.perf_counter() - start
                
                # Step 2: Search for relevant chunks
                query_status.markdown("<small>**[2/4]** Searching database...</small>", unsafe_allow_html=True)
                start = time.perf_counter()
                retrieved_texts, num_results = endee_client.search_vectors(query_vector, top_k=6)
                query_times["Vector Search"] = time.perf_counter() - start
                
                if not retrieved_texts:
                    query_status.error("No chunk context-file found, Please re-index the document")
                    # query_status.markdown("No", unsafe_allow_html=True)
                    with st.chat_message("assistant"):
                        st.error("Cached chunk context not available. Please re-index the document to regenerate cache.")

                    raise RuntimeError("No context") 
                    
                
                # Step 3: Prepare context with chat history
                query_status.markdown("<small>**[3/4]** Preparing context...</small>", unsafe_allow_html=True)
                
                # Build conversation history for context
                conversation_context = ""
                if len(st.session_state.chat_history) > 0:
                    conversation_context = "\n\nPrevious conversation:\n"
                    # Include last 3 exchanges for context
                    for chat in st.session_state.chat_history[-3:]:
                        conversation_context += f"Q: {chat['question']}\nA: {chat['answer']}\n\n"
                
                document_context = "\n\n".join(retrieved_texts[:4])
                
                # Step 4: Generate answer with memory
                query_status.markdown("<small>**[4/4]** Generating response...</small>", unsafe_allow_html=True)


                # Prompt
                prompt = f"""
                    You are an AI assistant helping users understand a document. Answer the question based on the provided context and conversation history.

                    Document Context:
                    {document_context}
                    {conversation_context}

                    Current Question:
                    {question}

                    Instructions:
                    - Answer based on the document context provided
                    - Reference previous conversation if relevant to provide continuity
                    - If the answer isn't in the context, say so clearly
                    - Be concise but thorough

                    Answer:
                    """
                start = time.perf_counter()

                response = rag_system.client.models.generate_content(
                    model=rag_system.model_name,
                    contents=prompt
                )

                query_times["LLM Generation"] = time.perf_counter() - start
                query_times["Total Query Time"] = time.perf_counter() - query_start_total
                answer = response.text
                
                # Clear query status
                query_status.empty()
                
                # Display answer
                with st.chat_message("assistant"):
                    st.write(answer)
                
                # Save to chat history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "chunks_used": len(retrieved_texts[:4])
                })
                
                # Update sidebar with query stats
                st.sidebar.markdown("---")
                st.sidebar.markdown("#### LAST QUERY STATS")
                st.sidebar.markdown(f"""
                <small>
                
                | Metric | Value |
                |--------|-------|
                | Vectors Retrieved | {num_results} |
                | Chunks Used | {len(retrieved_texts[:4])} |
                | Chat History | {len(st.session_state.chat_history)} exchanges |
                | Context Aware | Yes |
                
                </small>
                """, unsafe_allow_html=True)

                st.sidebar.markdown("---")
                st.sidebar.markdown("#### ⏱️ QUERY PERFORMANCE")

                st.sidebar.markdown("<small>", unsafe_allow_html=True)

                for step, duration in query_times.items():
                    st.sidebar.markdown(
                        f"<small>• {step}: `{duration:.2f}s`</small>",
                        unsafe_allow_html=True
                    )

                st.sidebar.markdown("</small>", unsafe_allow_html=True)
                

                # Option to view retrieved chunks
                with st.expander(f"📄 Context Used ({len(retrieved_texts[:4])} chunks)", expanded=False):
    
                    st.markdown(
                        """
                        <style>
                        .chunk-box {
                            background-color: #3E424B;
                            padding: 8px 12px;
                            border-radius: 8px;
                            margin-bottom: 8px;
                            font-size: 0.82rem;
                            line-height: 1.3;
                            max-height: 140px;
                            overflow-y: auto;
                            border: 1px solid #e0e0e0;
                        }
                        .chunk-title {
                            font-size: 0.75rem;
                            font-weight: 600;
                            margin-bottom: 4px;
                            color: #555;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )

                    for i, text in enumerate(retrieved_texts[:4]):
                        preview = text[:400] + "..." if len(text) > 400 else text
                        
                        st.markdown(
                            f"""
                            <div class="chunk-title">Chunk {i+1}</div>
                            <div class="chunk-box">
                            {preview}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            except RuntimeError:
                pass
            except Exception as e:
                query_status.error(f"Error: {str(e)}")
                with st.chat_message("assistant"):
                    st.error(f"An error occurred: {str(e)}")
                

# Show all indexed PDFs in sidebar
# st.sidebar.markdown("---")
# st.sidebar.markdown("#### INDEXED DOCUMENTS")
# try:
#     if db_active:
#         all_indexes = endee_client.list_all_indexes()
#         if all_indexes and "indexes" in all_indexes:
#             # pdf_indexes = [idx for idx in all_indexes["indexes"] if idx["name"].startswith("pdf_rag_")]
#             pdf_indexes = [idx for idx in all_indexes["indexes"] if idx["name"].startswith("rag_")]
#             if pdf_indexes:
#                 st.sidebar.markdown("<small>", unsafe_allow_html=True)
#                 for idx in pdf_indexes:
#                     idx_name = idx["name"]
#                     # Extract PDF name from index name
#                     pdf_name = idx_name.replace("pdf_rag_", "").replace("_", " ")
#                     is_current = idx_name == st.session_state.current_index
#                     status = "● " if is_current else "○ "
#                     st.sidebar.markdown(f"<small>{status}`{pdf_name[:25]}`</small>", unsafe_allow_html=True)
#                 st.sidebar.markdown("</small>", unsafe_allow_html=True)
#             else:
#                 st.sidebar.markdown("<small>No documents indexed</small>", unsafe_allow_html=True)
#     else:
#             st.sidebar.markdown("<small>Database offline</small>", unsafe_allow_html=True)
# except Exception as e:
#     st.sidebar.markdown("<small>Unable to load index list</small>", unsafe_allow_html=True)