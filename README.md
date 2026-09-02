# Advanced Multi-PDF RAG System

An advanced Retrieval-Augmented Generation (RAG) system for querying multiple research papers using structure-aware document processing, hybrid retrieval, reranking, conversational memory, and RAGAS evaluation.

The system processes multiple PDF research papers, converts their content into searchable chunks, retrieves the most relevant information using both semantic and lexical search, reranks the results, and generates grounded answers using Azure OpenAI.

---

## 1. Project Overview

This project implements an end-to-end Multi-PDF RAG pipeline.

The system supports:

- Multiple PDF documents
- Structure-aware PDF extraction using Docling
- Structure-aware chunking
- BGE-M3 embeddings
- FAISS vector search
- BM25 keyword search
- Reciprocal Rank Fusion (RRF)
- BGE Reranker
- Azure OpenAI GPT-5.4-mini
- Conversational memory
- Source/page attribution
- Gradio user interface
- RAGAS evaluation

### Research Papers

The system is designed to work with the following research papers:

1. **Attention Is All You Need**
   - arXiv: 1706.03762

2. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**
   - arXiv: 1810.04805

3. **Language Models are Few-Shot Learners**
   - arXiv: 2005.14165

4. **RoBERTa: A Robustly Optimized BERT Pretraining Approach**
   - arXiv: 1907.11692

5. **XLNet: Generalized Autoregressive Pretraining for Language Understanding**
   - arXiv: 1910.10683

---

# 2. Architecture

The overall RAG pipeline is:

PDF
↓
Docling PDF Extraction
↓
Structure-Aware Chunking
↓
BGE-M3 Embeddings
↓
Unified FAISS Vector Database
↓
┌─────────────────────┐
│                     │
│ Dense Retrieval     │
│ FAISS               │
│                     │
└──────────┬──────────┘
           │
           ├──────────────┐
           │              │
           ↓              ↓
      BM25 Retrieval   FAISS Retrieval
           │              │
           └──────┬───────┘
                  ↓
          RRF Hybrid Fusion
                  ↓
          BGE Reranker v2-M3
                  ↓
        Top Relevant Context
                  ↓
       Prompt + Conversation
            History
                  ↓
       Azure OpenAI GPT-5.4-mini
                  ↓
             Final Answer
                  ↓
          Source References
  ---

# 3. Setup and Run the Application

## 3.1 Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate

## 3.2 Install Dependencies
pip install -r requirements.txt

## 3.3 Run the Application
python ui.py
follow the http link to open the Gradio UI.
