# Implementing-Agentic-Retrieval-Augmented-Generation-RAG-
Implementing Agentic Retrieval Augmented Generation (RAG) using Langchain:
In Native RAG the user query is fed into the RAG pipeline which does retrieval, reranking, synthesis and generates a response.

**What is Agentic RAG ?**
Agentic RAG is an agent based approach to perform question answering over multiple documents in an orchestrated fashion, to compare different documents, summarise a specific document or compare various summaries. Agentic RAG is a flexible approach and framework to question answering.Here we essentially use agents instead of a LLM directly to accomplish a set of tasks which requires planning, multi step reasoning, tool use and/or learning over time.

**Basic Architecture**:
The basic architecture is to setup a document agent of each of the documents, with each document agent being able to perform question answering and summarisation within its own document.
Then a top level agent (meta-agent) is setup managing all of the lower order document agents.

**Technology Stack Used**
Langchain — more specifically LCEL : Orchestration framework to develop LLM applications
OpenAI — LLM
FAISS-cpu — Vectorstore

**Data Sources**
Here we will leverage ArxivLoader to retrieve metadata of articles published on arXiv.

Please see attached python file in this repository for code implementation.
