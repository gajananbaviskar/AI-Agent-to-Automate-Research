# 🚀 Loacal-AI-Agent-to-Automate-Research

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-green)
![Ollama](https://img.shields.io/badge/LLM-Llama_3-orange)
![AsyncIO](https://img.shields.io/badge/Concurrency-AsyncIO-red)

A high-performance, privacy-first **Retrieval-Augmented Generation (RAG)** agent built from first principles.

This project automates web research by asynchronously scraping multiple sources, embedding text into a local Vector Database (FAISS), and synthesizing answers using a local Large Language Model (Llama 3 via Ollama). It runs 100% locally with **zero API costs** and **zero data latency**.

---

## 🌟 Key Features

* **⚡ Blazing Fast:** Uses `AsyncIO` and `aiohttp` to scrape 10+ websites concurrently (in parallel) rather than sequentially.
* **🔒 Privacy-First:** All processing (Embedding, Indexing, Generation) happens locally on the device. No data is sent to OpenAI or cloud providers.
* **🧠 Vector Memory:** Uses **FAISS (Facebook AI Similarity Search)** for millisecond-latency vector retrieval.
* **🛡️ Robust Engineering:** Includes socket-level hard timeouts and "zombie connection" protection to prevent script freezes.
* **🚫 No LangChain:** Built from scratch using native libraries to demonstrate a deep understanding of RAG architecture without relying on abstraction frameworks.

---

## 🏗️ Architecture & Workflow

The agent follows a strict **Retrieval-Augmented Generation (RAG)** pipeline:

1.  **Search:** Queries DuckDuckGo to find relevant URLs.
2.  **Concurrent Scraping (AsyncIO):** Fires parallel requests to fetch all pages instantly.
3.  **Chunking & Embedding:** Splits text into 120-word sliding windows and converts them to vectors using `sentence-transformers/all-MiniLM-L6-v2`.
4.  **Vector Indexing (FAISS):** Stores vectors in a RAM-based `IndexFlatIP`.
    * *Optimization:* Vectors are normalized so that Inner Product (Dot Product) is mathematically equivalent to Cosine Similarity, but faster.
5.  **Ranking:** Retrieves the top 7 most relevant text chunks based on semantic similarity to the user's query.
6.  **Generation:** Feeds the ranked context to **Llama 3** (via Ollama) to synthesize a final, cited answer.

---

## 🛠️ The Engineering Journey (Challenges & Evolution)

This project evolved through several iterations to solve specific engineering bottlenecks.

### Phase 1: The "Manual" Script
* **Approach:** Sequential scraping + Manual NumPy Cosine Similarity.
* **Problem:** It wasn't "Generative." It could only extract sentences, not synthesize answers. It was also slow (blocking I/O).

### Phase 2: Adding Intelligence
* **Approach:** Integrated **Ollama (Llama 3)**.
* **Problem:** **Latency & Amnesia.** The model had to re-scrape and re-embed data for every single query. It was computationally wasteful.

### Phase 3: The "Database Lock" Challenge
* **Approach:** Added **ChromaDB** for persistent storage.
* **The Bug:** The script suffered from "Database Locks" if it crashed midway. Additionally, "Zombie Servers" (connections that stay open but send no data) caused the script to freeze indefinitely.

### Phase 4: The Stability Fix (FAISS + Sockets)
* **Solution:** Switched to **FAISS** (RAM-only) to eliminate file locks. Implemented **Socket Hard-Timeouts** (Tuple-based: 3s connect, 10s read) to aggressively kill zombie connections.
* **Result:** A crash-proof system that "self-heals" when encountering bad URLs.

### Phase 5: The Speed Upgrade (AsyncIO) 🚀
* **Solution:** Refactored the scraping engine from `requests` (Sync) to `aiohttp` (Async).
* **Result:** Reduced total execution time for 10 URLs from **~25 seconds** to **~3 seconds**. The CPU no longer waits for network responses; it processes concurrent streams.

---

## 💻 Technical Stack

* **Language:** Python
* **Concurrency:** `asyncio`, `aiohttp`
* **Vector Search:** `faiss-cpu` (IndexFlatIP)
* **Embeddings:** `sentence-transformers` (all-MiniLM-L6-v2)
* **Inference:** `ollama` (running Llama 3)
* **Search:** `duckduckgo-search`
* **Parsing:** `beautifulsoup4`

---

## 💡 Design Decisions (Why this Stack?)

Every component in this architecture was chosen to balance speed, privacy, and computational efficiency on edge hardware (standard laptops).

| Component | Choice | Reasoning |
| :--- | :--- | :--- |
| **Generative Model** | **Llama 3 (via Ollama)** | Chosen for its state-of-the-art performance in the 8B parameter class. It runs efficiently on local hardware/CPU via Ollama quantization, ensuring **100% data privacy** (critical for enterprise use cases) and zero inference cost. |
| **Embedding Model** | **all-MiniLM-L6-v2** | A "tiny giant" in the Sentence Transformers library. It maps text to a 384-dimensional vector space. We chose this over larger models (like BERT) because it offers the **best speed-to-accuracy ratio**, allowing for near-instant embedding on a CPU. |
| **Vector Database** | **FAISS (IndexFlatIP)** | We replaced ChromaDB with FAISS to solve "database locking" issues. FAISS runs entirely in RAM using C++ optimization. We use `IndexFlatIP` (Inner Product) with normalized vectors, which is mathematically equivalent to **Cosine Similarity** but significantly faster to compute. |
| **Concurrency** | **AsyncIO + aiohttp** | Sequential scraping (Looping `requests.get`) creates massive latency bottlenecks. By moving to an Event Loop model, we fire all network requests simultaneously, reducing the total wait time from $T \times N$ to $max(T)$. |
| **Search Provider** | **DuckDuckGo (DDGS)** | chosen to avoid the complexity and rate limits of paid APIs (like Bing/Google Custom Search). It allows for rapid prototyping and open-source distribution without requiring users to generate API keys. |

---

## 🚀 How to Run

### Prerequisites
1.  **Install Ollama:** Download from [ollama.com](https://ollama.com).
2.  **Pull the Model:**
    ```bash
    ollama pull llama3
    ```

### Installation
1.  Clone the repo:
    ```bash
    git clone https://github.com/gajananbaviskar/AI-Agent-to-Automate-Research.git
    cd local-rag-agent
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage
Run the script directly:
```bash
python AI Agent to Automate Research.py
```
Modify the q variable in the __main__ block to ask different questions.

### 📄 License
MIT License. Free to use for educational and personal research.

### Reference and Inspration
https://amanxai.com/2025/11/11/build-an-ai-agent-to-automate-your-research/
