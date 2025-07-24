import streamlit as st
import os
import platform
from dotenv import load_dotenv
from PIL import Image
import fitz
import asyncio
from rag_utils import PDFRAGSystem

load_dotenv()

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

if "rag" not in st.session_state:
    st.session_state.rag = PDFRAGSystem()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None
if "pdf_images" not in st.session_state:
    st.session_state.pdf_images = []
if "language" not in st.session_state:
    st.session_state.language = "English"

def render_pdf_as_images(file_path, dpi=150):
    """Convert all PDF pages to images"""
    try:
        doc = fitz.open(file_path)
        images = []
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        doc.close()
        if len(images) > 50:  # Warn for large PDFs
            st.warning(f"Loaded {len(images)} pages. Large PDFs may slow down the app.")
        return images
    except Exception as e:
        st.error(f"Error rendering PDF: {str(e)}")
        return []

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temp directory and clean up old files"""
    try:
        if not os.path.exists("temp"):
            os.makedirs("temp")
        for old_file in os.listdir("temp"):
            os.remove(os.path.join("temp", old_file))
        file_path = os.path.join("temp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

with st.sidebar:
    st.title("📚 PDF RAG Settings")
    
    uploaded_file = st.file_uploader("📂 Upload PDF", type=["pdf"])
    
    st.session_state.language = st.selectbox(
        "🌐 Select Response Language",
        ["English", "Hindi", "Spanish", "French", "German"],
        index=0
    )
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    with st.expander("ℹ️ Help & Tips"):
        st.markdown("""
        - Upload a PDF to start. Note: Large PDFs may take time to load all pages.
        - Ask **direct questions** (e.g., "What is the main topic?").
        - Try **indirect questions** (e.g., "How can the strategies apply to a new market?").
        - For **case studies**, ask for analysis or improvements (e.g., "Analyze the case study and suggest solutions").
        - Confidence score (0-100%) shows answer alignment with the PDF.
        - Error risk: Type 1 (High Risk), Type 2 (Partial), None (Reliable).
        """)

col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.header("📄 PDF Viewer")
    if st.session_state.current_pdf:
        with st.container(height=700):
            for i, img in enumerate(st.session_state.pdf_images):
                st.image(img, use_container_width=True, caption=f"Page {i+1}")
                st.write("")
    else:
        st.info("Upload a PDF to view it here")

with col2:
    st.title("💬 PDF RAG Assistant")
    
    if uploaded_file and uploaded_file.name != st.session_state.current_pdf:
        with st.spinner("Processing PDF..."):
            file_path = save_uploaded_file(uploaded_file)
            if file_path:
                st.session_state.pdf_images = render_pdf_as_images(file_path, dpi=150)
                success, message = st.session_state.rag.load_pdf(file_path)
                if success:
                    st.session_state.current_pdf = uploaded_file.name
                    st.session_state.messages.append({
                        "role": "system",
                        "content": f"📄 Now analyzing: {uploaded_file.name}"
                    })
                    st.rerun()
                else:
                    st.error(message)
    
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"]["answer"] if isinstance(message["content"], dict) else message["content"])
                    if isinstance(message["content"], dict):
                        with st.expander("🔍 Confidence Details"):
                            st.write(f"Confidence: {message['content']['confidence']}")
                            st.write(f"Error Risk: {message['content']['error_risk']}")
                            st.write("Supporting Context:")
                            for snippet in message['content']['context_snippets']:
                                st.caption(snippet)
    
    if prompt := st.chat_input(f"Ask a question in {st.session_state.language} (e.g., direct, indirect, or case study)", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        
        with st.spinner("🤔 Thinking..."):
            if platform.system() == "Emscripten":
                async def get_response():
                    return await asyncio.get_event_loop().run_in_executor(
                        None, st.session_state.rag.ask_question, prompt, st.session_state.language
                    )
                response = asyncio.run(get_response())
            else:
                response = st.session_state.rag.ask_question(prompt, st.session_state.language)
        
        with chat_container:
            with st.chat_message("assistant"):
                if "error" in response:
                    st.error(response["error"])
                else:
                    st.markdown(response["answer"])
                    with st.expander("🔍 Confidence Details"):
                        st.write(f"Confidence: {response['confidence']}")
                        st.write(f"Error Risk: {response['error_risk']}")
                        st.write("Supporting Context:")
                        for snippet in response["context_snippets"]:
                            st.caption(snippet)
        
        st.session_state.messages.append({"role": "assistant", "content": response})