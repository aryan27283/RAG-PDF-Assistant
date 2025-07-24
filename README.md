

Process of Rag:
-Text Chunks → Embeddings
OllamaEmbeddings converts PDF text into vectors.
Stored in ChromaDB (a vector database).

-Retrieval Phase
When you ask a question, its embedding is compared with stored vectors.
The closest-matching chunks (by cosine similarity) are retrieved.

-Generation Phase
The LLM (llama3) reads the retrieved chunks to generate an answer.



 Features
🔒 Fully local processing - no data leaves your machine
📄 PDF processing with intelligent chunking
🧠 Multi-query retrieval for better context understanding
🎯 Advanced RAG implementation using LangChain
🖥️ Clean Streamlit interface
 





