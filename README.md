# Local-RAG-Assistant

A complete Retrieval-Augmented Generation (RAG) pipeline built using LangChain, ChromaDB, Sentence Transformers, and Ollama to run Large Language Models completely locally without requiring any external API keys.

This project demonstrates how modern AI assistants work internally by combining:

* Semantic Search
* Vector Databases
* Embeddings
* Local LLM Inference
* Context-Aware Question Answering

The system can ingest custom PDF documents, split them into semantic chunks, generate embeddings, store them in a persistent ChromaDB vector database, retrieve relevant context using similarity search, and generate grounded responses using locally running LLMs through Ollama.

## Features

* PDF document ingestion pipeline
* Semantic text chunking
* Embedding generation using Sentence Transformers
* Persistent vector storage using ChromaDB
* Similarity-based retrieval
* Local LLM execution using Ollama
* LangChain integration
* Context-aware question answering
* Fully local setup with no API dependency

## Tech Stack

* Python
* LangChain
* Ollama
* ChromaDB
* Sentence Transformers
* Qwen3 / Llama3.2
* PyPDFLoader

## Architecture

PDF Documents
→ Text Chunking
→ Embedding Generation
→ ChromaDB Vector Store
→ Semantic Retrieval
→ Local LLM (Ollama)
→ Answer Generation


This project helped build a deep understanding of:

* Retrieval-Augmented Generation (RAG)
* Embedding models
* Vector databases
* Prompt engineering
* Local LLM deployment
* AI assistant architecture
