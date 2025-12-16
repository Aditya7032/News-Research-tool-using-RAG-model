This project is an AI-powered news and document research tool that combines traditional information retrieval techniques with modern NLP and Retrieval-Augmented Generation (RAG) to enable efficient document analysis and question answering.

The system supports PDF, DOCX, TXT files, live news fetching via NewsAPI and Google Search, and performs hybrid ranking using BM25 and TF-IDF. For semantic understanding, documents are chunked and embedded using Sentence-BERT, enabling vector similarity search across content.

A RAG pipeline retrieves the most relevant document chunks and generates grounded summaries and answers, reducing hallucinations and improving reliability. The tool also includes few-shot Q&A, agentic multi-step analysis, cross-document semantic search, lightweight knowledge graph construction, and named entity recognition (NER) for deeper insights.

Built with Python and Streamlit, this project demonstrates a practical and scalable approach to AI-driven research, news intelligence, and document understanding.
