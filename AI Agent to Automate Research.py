#with AsyncIO + FAISS + Ollama
import re
import urllib.parse
from ddgs import DDGS
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import time
import ollama
import faiss
import asyncio  # <--- NEW: The Concurrency Engine
import aiohttp  # <--- NEW: The Async HTTP Client

# --- Configuration ---
SEARCH_RESULTS = 10
PASSAGES_PER_PAGE = 10
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_PASSAGES = 7
TIMEOUT = 10

def unwrap_ddg(url):
    """Handles DuckDuckGo redirect URLs."""
    try:
        parsed = urllib.parse.urlparse(url)
        if "duckduckgo.com" in parsed.netloc:
            qs = urllib.parse.parse_qs(parsed.query)
            uddg = qs.get("uddg")
            if uddg: return urllib.parse.unquote(uddg[0])
    except Exception: pass
    return url

def search_web(query, max_results=SEARCH_RESULTS):
    """Searches the web (Sync operation is fine here)."""
    urls = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                url = r.get("href") or r.get("url")
                if not url: continue
                url = unwrap_ddg(url)
                urls.append(url)
    except Exception as e:
        print(f"Search Warning: {e}")
    return urls

# --- NEW ASYNC FUNCTIONS ---

async def fetch_text_async(session, url):
    """
    Asynchronously fetches and cleans text from a URL.
    Uses the shared 'session' to speed up connections.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        # TIMEOUT: Connect in 4s, Read entire page in 10s
        timeout = aiohttp.ClientTimeout(connect=4, total=10)
        
        async with session.get(url, headers=headers, timeout=timeout) as response:
            if response.status != 200:
                return None
            
            html = await response.text()
            
            # CPU-bound cleaning (Fast enough to keep in main thread for simple scripts)
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "iframe", "nav", "aside"]):
                tag.extract()
            
            paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            text = " ".join([p for p in paragraphs if p])
            
            if text.strip():
                clean_text = re.sub(r"\s+", " ", text).strip()
                return {"url": url, "text": clean_text}
                
    except Exception:
        # Fail silently on individual timeouts so others keep going
        return None
    return None

def chunk_passages(text, max_words=120):
    words = text.split()
    if not words: return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + max_words]
        chunks.append(" ".join(chunk))
        i += max_words
    return chunks

class ShortResearchAgent:
    def __init__(self, embed_model=EMBEDDING_MODEL):
        print(f"Loading embedder: {embed_model}...", flush=True)
        self.embedder = SentenceTransformer(embed_model)
        
        # FAISS Setup
        self.dimension = 384
        self.index = faiss.IndexFlatIP(self.dimension)
        self.stored_docs = []

    # Wrapper to run async code from sync main
    def run(self, query):
        return asyncio.run(self.process_async(query))

    async def process_async(self, query):
        start = time.time()
        
        # Clean RAM
        self.index.reset()
        self.stored_docs = []
        
        # 1. Search
        print(f"Searching for: {query}...", flush=True)
        # Search is sync, but it's fast enough
        urls = search_web(query)
        print(f"Found {len(urls)} urls. Starting parallel scrape...", flush=True)
        
        # 2. Async Fetching (The Magic Step)
        async with aiohttp.ClientSession() as session:
            tasks = []
            for u in urls:
                print(f"  - Queued: {u[:40]}...")
                tasks.append(fetch_text_async(session, u))
            
            # FIRE ALL REQUESTS AT ONCE
            results = await asyncio.gather(*tasks)
        
        # 3. Processing Results
        valid_results = [r for r in results if r]
        print(f"\nSuccessfully scraped {len(valid_results)} pages.", flush=True)
        
        if not valid_results:
            return {"query": query, "passages": [], "summary": "No data found.", "references": [], "time": time.time()-start}

        # Indexing loop
        for item in valid_results:
            url = item['url']
            txt = item['text']
            
            chunks = chunk_passages(txt, max_words=120)
            if not chunks: continue
            
            # Embed & Add to FAISS
            embeddings = self.embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
            self.index.add(embeddings)
            
            for c in chunks:
                self.stored_docs.append({"url": url, "passage": c})
        
        print(f"Total Database Size: {self.index.ntotal} passages.", flush=True)

        # 4. Rank
        q_emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(q_emb, TOP_PASSAGES)
        
        top_passages = []
        for i in range(len(I[0])):
            idx = I[0][i]
            score = D[0][i]
            if idx < len(self.stored_docs):
                doc = self.stored_docs[idx]
                top_passages.append({
                    "url": doc["url"],
                    "passage": doc["passage"],
                    "score": float(score)
                })

        # 5. Generation
        print("Preparing context for Llama 3...", flush=True)
        context_text = ""
        references = []
        url_to_id = {}
        
        for tp in top_passages:
            url = tp['url']
            if url not in url_to_id:
                url_to_id[url] = len(references) + 1
                references.append(url)
            ref_id = url_to_id[url]
            context_text += f"{tp['passage']} [{ref_id}]\n"

        system_prompt = """"You are a research assistant. Answer the user's question using ONLY the provided context. 
        Cite your sources using the numbers provided (e.g., [1]). The response should contain atleast 5 bullet points.
        Use more if the current bullet points in the answer do not cover the whole depth and the intent of the question asked.
        If the context doesn't contain the answer, say you don't know. Format the output as a compact list with NO empty lines between bullet points.
        Always arrange and sort the bullet points point according to their related topics"""
        
        user_prompt = f"Question: {query}\n\nContext:\n{context_text}"

        print("Generating answer with Local Ollama...", flush=True)
        try:
            # Note: Ollama client is sync, which blocks the loop. 
            # In a massive production app, we'd run this in a thread, 
            # but for a portfolio script, this is perfectly fine.
            response = ollama.chat(model='llama3', messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ])
            summary = response['message']['content']
        except Exception as e:
            summary = f"Error connecting to Ollama: {e}"

        elapsed = time.time() - start
        return {
            "query": query, 
            "passages": top_passages, 
            "summary": summary, 
            "references": references, 
            "time": elapsed
        }

if __name__ == "__main__":
    agent = ShortResearchAgent()
    q = "What are the dangers and harms of AI?"
    
    print(f"Running query: {q}\n", flush=True)
    out = agent.run(q)
    
    print("\nTop passages:", flush=True)
    for p in out["passages"]:
        print(f"- Score {p['score']:.3f} src {p['url']}\n  {p['passage'][:50]}...\n")
        
    print("\n--- Research Summary ---", flush=True)
    print(out["summary"])
    
    print("\n--- References ---", flush=True)
    if out["references"]:
        for i, ref in enumerate(out["references"]):
            print(f"[{i+1}] {ref}")
    else:
        print("No references found.")
        
    print("--------------------------", flush=True)
    print(f"\nDone in {out['time']:.1f}s")

























### ALL THE OLDER VERSIONS OF THE PROJECT




# #with FAISS + Ollama
# import re
# import urllib.parse
# from ddgs import DDGS
# import requests
# from bs4 import BeautifulSoup
# from sentence_transformers import SentenceTransformer
# import time
# import ollama 
# import faiss
# import socket  # <--- CRITICAL: Prevents "Zombie" freezes

# # --- Configuration ---
# SEARCH_RESULTS = 10       
# PASSAGES_PER_PAGE = 10    
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# TOP_PASSAGES = 10          
# TIMEOUT = 10              

# def unwrap_ddg(url):
#     try:
#         parsed = urllib.parse.urlparse(url)
#         if "duckduckgo.com" in parsed.netloc:
#             qs = urllib.parse.parse_qs(parsed.query)
#             uddg = qs.get("uddg")
#             if uddg: return urllib.parse.unquote(uddg[0])
#     except Exception: pass
#     return url

# def search_web(query, max_results=SEARCH_RESULTS):
#     urls = []
#     # Hard timeout: Kill any connection attempt that takes >10s
#     socket.setdefaulttimeout(10)
#     try:
#         with DDGS() as ddgs:
#             for r in ddgs.text(query, max_results=max_results):
#                 url = r.get("href") or r.get("url")
#                 if not url: continue
#                 url = unwrap_ddg(url) 
#                 urls.append(url)
#     except Exception as e:
#         print(f"Search Warning: {e}")
#     return urls

# def fetch_text(url):
#     # Browser user-agent to avoid being blocked
#     headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
#     try:
#         # CRITICAL FIX: Tuple timeout (Connect in 3s, Read in 10s)
#         # This prevents the "Freeze" you saw earlier.
#         r = requests.get(url, timeout=(3, 10), headers=headers, allow_redirects=True)
#         if r.status_code != 200: return ""
        
#         soup = BeautifulSoup(r.text, "html.parser")
#         for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "iframe", "nav", "aside"]):
#             tag.extract()
            
#         paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
#         text = " ".join([p for p in paragraphs if p])
        
#         if text.strip(): return re.sub(r"\s+", " ", text).strip()
            
#     except Exception: return "" 
#     return ""

# def chunk_passages(text, max_words=120):
#     words = text.split()
#     if not words: return []
#     chunks = []
#     i = 0
#     while i < len(words):
#         chunk = words[i : i + max_words]
#         chunks.append(" ".join(chunk))
#         i += max_words
#     return chunks

# class ShortResearchAgent:
#     def __init__(self, embed_model=EMBEDDING_MODEL):
#         print(f"Loading embedder: {embed_model}...", flush=True)
#         self.embedder = SentenceTransformer(embed_model)
        
#         # FAISS Setup
#         self.dimension = 384 
#         # IndexFlatIP + Normalized Vectors = Cosine Similarity
#         self.index = faiss.IndexFlatIP(self.dimension)
#         self.stored_docs = [] 

#     def run(self, query):
#         start = time.time()
        
#         # Reset memory for new query
#         self.index.reset()
#         self.stored_docs = []
        
#         # 1. Search
#         print(f"Searching for: {query}...", flush=True)
#         urls = search_web(query)
#         print(f"Found {len(urls)} urls.", flush=True)
        
#         # 2. Fetch & Chunk
#         print("Scraping websites...", flush=True)
#         for u in urls:
#             print(f"  - Fetching: {u[:40]}...", end="", flush=True) 
#             txt = fetch_text(u)
#             if not txt: 
#                 print(" [Failed/Skipped]", flush=True)
#                 continue
            
#             chunks = chunk_passages(txt, max_words=120)
#             if not chunks: 
#                 print(" [No Text]", flush=True)
#                 continue
            
#             # 3. Embed & Index
#             # Important: Normalize to make Inner Product act like Cosine Similarity
#             embeddings = self.embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
#             self.index.add(embeddings)
            
#             for c in chunks:
#                 self.stored_docs.append({"url": u, "passage": c})
#             print(" [Indexed]", flush=True)
        
#         if self.index.ntotal == 0:
#             return {"query": query, "passages": [], "summary": "No data found.", "references": [], "time": time.time()-start}
        
#         print(f"Total Database Size: {self.index.ntotal} passages.", flush=True)

#         # 4. Rank
#         q_emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
#         D, I = self.index.search(q_emb, TOP_PASSAGES)
        
#         top_passages = []
#         for i in range(len(I[0])):
#             idx = I[0][i]
#             score = D[0][i]
#             if idx < len(self.stored_docs):
#                 doc = self.stored_docs[idx]
#                 top_passages.append({
#                     "url": doc["url"],
#                     "passage": doc["passage"],
#                     "score": float(score)
#                 })

#         # 5. Generation
#         print("Preparing context for Llama 3...", flush=True)
#         context_text = ""
#         references = []
#         url_to_id = {}
        
#         for tp in top_passages:
#             url = tp['url']
#             if url not in url_to_id:
#                 url_to_id[url] = len(references) + 1
#                 references.append(url)
#             ref_id = url_to_id[url]
#             context_text += f"{tp['passage']} [{ref_id}]\n\n"

#         system_prompt = """You are a research assistant. Answer the user's question using ONLY the provided context. 
#         Cite your sources using the numbers provided (e.g., [1]). The response should contain atleast 5 bullet points.
#         Use more if the current bullet points in the answer do not cover the whole depth and the intent of the question asked.
#         If the context doesn't contain the answer, say you don't know. Format the output as a compact list with NO empty lines between bullet points.
#         Always arrange and sort the bullet points point according to their related topics"""
        
#         user_prompt = f"Question: {query}\n\nContext:\n{context_text}"

#         print("Generating answer with Local Ollama...", flush=True)
#         try:
#             # Disable socket timeout for generation (LLMs take time)
#             socket.setdefaulttimeout(None)
#             response = ollama.chat(model='llama3', messages=[
#                 {'role': 'system', 'content': system_prompt},
#                 {'role': 'user', 'content': user_prompt},
#             ])
#             summary = response['message']['content']
#         except Exception as e:
#             summary = f"Error connecting to Ollama: {e}"

#         elapsed = time.time() - start
#         return {
#             "query": query, 
#             "passages": top_passages, 
#             "summary": summary, 
#             "references": references, 
#             "time": elapsed
#         }

# if __name__ == "__main__":
#     agent = ShortResearchAgent()
#     q = "What is the shape of earth?"
    
#     print(f"Running query: {q}\n", flush=True)
#     out = agent.run(q)
    
#     print("\nTop passages:", flush=True)
#     for p in out["passages"]:
#         print(f"- Score {p['score']:.3f} src {p['url']}\n  {p['passage'][:50]}...\n")
        
#     print("\n--- Research Summary ---", flush=True)
#     print(out["summary"])
    
#     print("\n--- References ---", flush=True)
#     if out["references"]:
#         for i, ref in enumerate(out["references"]):
#             print(f"[{i+1}] {ref}")
#     else:
#         print("No references found.")
        
#     print("--------------------------", flush=True)
#     print(f"\nDone in {out['time']:.1f}s")








# # 2nd Update : Used ollama instead of all-MiniLM-L6-v2 for geenration
# import re
# import urllib.parse
# from ddgs import DDGS          # package name is 'ddgs' (duckduckgo_search renamed)
# import requests
# from bs4 import BeautifulSoup
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import time
# import ollama  # <--- NEW IMPORT
# import chromadb # <--- NEW: Import the Database

# SEARCH_RESULTS = 10       # How many URLs to check
# PASSAGES_PER_PAGE = 10    # How many passages to pull from each URL
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # Fast, high-quality model
# TOP_PASSAGES = 7          # How many relevant passages to use for the summary
# SUMMARY_SENTENCES = 20    # How many sentences in the final summary #not needed after 2nd update
# TIMEOUT = 10              # How long to wait for a webpage to load

# def unwrap_ddg(url):
#     """If DuckDuckGo returns a redirect wrapper, extract the real URL."""
#     try:
#         parsed = urllib.parse.urlparse(url)
#         if "duckduckgo.com" in parsed.netloc:
#             qs = urllib.parse.parse_qs(parsed.query)
#             uddg = qs.get("uddg")
#             if uddg:
#                 return urllib.parse.unquote(uddg[0])
#     except Exception:
#         pass
#     return url

# def search_web(query, max_results=SEARCH_RESULTS):
#     """Search the web and return a list of URLs."""
#     urls = []
#     with DDGS() as ddgs:
#         for r in ddgs.text(query, max_results=max_results):
#             url = r.get("href") or r.get("url")
#             if not url:
#                 continue
#             url = unwrap_ddg(url) # Clean up DDG redirect links
#             urls.append(url)
#     return urls

# def fetch_text(url, timeout=TIMEOUT):
#     """Fetch and clean text content from a URL."""
#     headers = {"User-Agent": "Mozilla/5.0 (research-agent)"}
#     try:
#         r = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
#         if r.status_code != 200:
#             return ""
#         ct = r.headers.get("content-type", "")
#         if "html" not in ct.lower(): # Skip non-HTML content
#             return ""
        
#         soup = BeautifulSoup(r.text, "html.parser")
        
#         # Remove all annyoing tags
#         for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "iframe", "nav", "aside"]):
#             tag.extract()
            
#         # Get all paragraph text
#         paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
#         text = " ".join([p for p in paragraphs if p])
        
#         if text.strip():
#             # Clean up whitespace
#             return re.sub(r"\s+", " ", text).strip()
            
#         # --- Fallback logic if <p> tags fail ---
#         meta = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
#         if meta and meta.get("content"):
#             return meta["content"].strip()
#         if soup.title and soup.title.string:
#             return soup.title.string.strip()
            
#     except Exception:
#         return "" # Fail silently
#     return ""

# def chunk_passages(text, max_words=120):
#     """Split long text into smaller passages."""
#     words = text.split()
#     if not words:
#         return []
#     chunks = []
#     i = 0
#     while i < len(words):
#         chunk = words[i : i + max_words]
#         chunks.append(" ".join(chunk))
#         i += max_words
#     return chunks

# def split_sentences(text):
#     """A simple sentence splitter."""
#     parts = re.split(r'(?<=[.!?])\s+', text)
#     return [p.strip() for p in parts if p.strip()]
  
# # 2nd Update : Used ollama instead of all-MiniLM-L6-v2 for geenration
# class ShortResearchAgent:
#     def __init__(self, embed_model=EMBEDDING_MODEL):
#         print(f"Loading embedder: {embed_model}...")
#         self.embedder = SentenceTransformer(embed_model)
#         # No API Key needed for Ollama!

#     def run(self, query):
#         start = time.time()
        
#         # 1. Search
#         urls = search_web(query)
#         print(f"Found {len(urls)} urls.")
        
#         # 2. Fetch & Chunk
#         docs = []
#         for u in urls:
#             txt = fetch_text(u)
#             if not txt: continue
#             chunks = chunk_passages(txt, max_words=120)
#             for c in chunks[:PASSAGES_PER_PAGE]:
#                 docs.append({"url": u, "passage": c})
        
#         if not docs:
#             return {"query": query, "passages": [], "summary": "No data found.", "references": []}
        
#         # 3. Embed
#         texts = [d["passage"] for d in docs]
#         emb_texts = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
#         q_emb = self.embedder.encode([query], convert_to_numpy=True)[0]
        
#         # 4. Rank
#         def cosine(a, b): 
#             return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
            
#         sims = [cosine(e, q_emb) for e in emb_texts]
#         top_idx = np.argsort(sims)[::-1][:TOP_PASSAGES]
#         top_passages = [{"url": docs[i]["url"], "passage": docs[i]["passage"], "score": float(sims[i])} for i in top_idx]

#         # 5. Generation (Local Llama 3)
#         print("preparing context for Llama 3...")
        
#         # Build Context
#         context_text = ""
#         references = []
#         url_to_id = {}
        
#         for tp in top_passages:
#             url = tp['url']
#             if url not in url_to_id:
#                 url_to_id[url] = len(references) + 1
#                 references.append(url)
#             ref_id = url_to_id[url]
#             context_text += f"{tp['passage']} [{ref_id}]\n\n"

#         # Construct Prompt
#         system_prompt = """You are a research assistant. Answer the user's question using ONLY the provided context. 
#         Cite your sources using the numbers provided (e.g., [1]). The response should contain atleast 5 bullet points.
#         Use more if the current bullet points in the answer do not cover the whole depth and the intent of the question asked.
#         If the context doesn't contain the answer, say you don't know."""
        
#         user_prompt = f"Question: {query}\n\nContext:\n{context_text}"

#         # Call Ollama Locally
#         print("Generating answer with Local Ollama (this may take a moment)...")
#         try:
#             response = ollama.chat(model='llama3', messages=[
#                 {'role': 'system', 'content': system_prompt},
#                 {'role': 'user', 'content': user_prompt},
#             ])
#             summary = response['message']['content']
#         except Exception as e:
#             summary = f"Error connecting to Ollama: {e}. Make sure the Ollama app is running."

#         elapsed = time.time() - start
#         return {
#             "query": query, 
#             "passages": top_passages, 
#             "summary": summary, 
#             "references": references, 
#             "time": elapsed
#         }


# # --- Main Execution Block ---
# if __name__ == "__main__":
#     agent = ShortResearchAgent()
#     q = "What are the dangers and harms of AI?"
    
#     print(f"Running query: {q}\n")
#     out = agent.run(q)
    
#     # Optional: Print raw passages (commented out to keep output clean)
#     print("\nTop passages:")
#     for p in out["passages"]:
#         print(f"- score {p['score']:.3f} src {p['url']}\n  {p['passage'][:50]}...\n")
        
#     print("\n--- Extractive Summary ---")
#     print(out["summary"])
    
#     print("\n--- References ---")
#     if out["references"]:
#         for i, ref in enumerate(out["references"]):
#             print(f"[{i+1}] {ref}")
#     else:
#         print("No references found.")
        
#     print("--------------------------")
#     print(f"\nDone in {out['time']:.1f}s")
















#FIRST CODE

# class ShortResearchAgent:
#     def __init__(self, embed_model=EMBEDDING_MODEL):
#         print(f"Loading embedder: {embed_model}...")
#         # This downloads the model on first run
#         self.embedder = SentenceTransformer(embed_model)

#     def run(self, query):
#         start = time.time()
        
#         # 1. Search
#         urls = search_web(query)
#         print(f"Found {len(urls)} urls.")
        
#         # 2. Fetch & Chunk
#         docs = []
#         for u in urls:
#             txt = fetch_text(u)
#             if not txt:
#                 continue
#             chunks = chunk_passages(txt, max_words=120)
#             for c in chunks[:PASSAGES_PER_PAGE]:
#                 docs.append({"url": u, "passage": c})
        
#         if not docs:
#             print("No documents fetched.")
#             return {"query": query, "passages": [], "summary": ""}
      
#         # 3. Embed (Turn text into numbers)
#         texts = [d["passage"] for d in docs]
#         emb_texts = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
#         q_emb = self.embedder.encode([query], convert_to_numpy=True)[0]
        
#         # 4. Rank (Find similarity)
#         def cosine(a, b): 
#             return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
            
#         sims = [cosine(e, q_emb) for e in emb_texts]
#         top_idx = np.argsort(sims)[::-1][:TOP_PASSAGES]
#         top_passages = [{"url": docs[i]["url"], "passage": docs[i]["passage"], "score": float(sims[i])} for i in top_idx]

#         # 5. Summarize (Extractive)
#         sentences = []
#         for tp in top_passages:
#             for s in split_sentences(tp["passage"]):
#                 sentences.append({"sent": s, "url": tp["url"]})
        
#         if not sentences:
#             summary = "No summary could be generated."
#         else:
#             sent_texts = [s["sent"] for s in sentences]
#             sent_embs = self.embedder.encode(sent_texts, convert_to_numpy=True, show_progress_bar=False)
#             sent_sims = [cosine(e, q_emb) for e in sent_embs]
            
#             top_sent_idx = np.argsort(sent_sims)[::-1][:SUMMARY_SENTENCES]
#             chosen = [sentences[idx] for idx in top_sent_idx]

#             # De-duplicate and format
#             seen = set()
#             lines = []
#             for s in chosen:
#                 key = s["sent"].lower()[:80] # Check first 80 chars for duplication
#                 if key in seen:
#                     continue
#                 seen.add(key)
#                 lines.append(f"{s['sent']} (Source: {s['url']})")
#             summary = " ".join(lines)

#         elapsed = time.time() - start
#         return {"query": query, "passages": top_passages, "summary": summary, "time": elapsed}

# if __name__ == "__main__":
#     agent = ShortResearchAgent()
#     q = "What are the etical dilemmas related to Retrieval-Augmented Generation RAG"
    
#     print(f"Running query: {q}\n")
#     out = agent.run(q)
    
#     print("\nTop passages:")
#     for p in out["passages"]:
#         print(f"- score {p['score']:.3f} src {p['url']}\n  {p['passage'][:200]}...\n")
        
#     print("--- Extractive summary ---")
#     print(out["summary"])
#     print("--------------------------")
#     print(f"\nDone in {out['time']:.1f}s")







# 1st Update : made it so that the references are not mentioned as a whole but as numbers like in resarch papers like this [1], [2], etc...
# class ShortResearchAgent:
#     def __init__(self, embed_model=EMBEDDING_MODEL):
#         print(f"Loading embedder: {embed_model}...")
#         self.embedder = SentenceTransformer(embed_model)

#     def run(self, query):
#         start = time.time()
        
#         # 1. Search
#         urls = search_web(query)
#         print(f"Found {len(urls)} urls.")
        
#         # 2. Fetch & Chunk
#         docs = []
#         for u in urls:
#             txt = fetch_text(u)
#             if not txt:
#                 continue
#             chunks = chunk_passages(txt, max_words=120)
#             for c in chunks[:PASSAGES_PER_PAGE]:
#                 docs.append({"url": u, "passage": c})
        
#         if not docs:
#             print("No documents fetched.")
#             return {"query": query, "passages": [], "summary": "", "references": []}
        
#         # 3. Embed
#         texts = [d["passage"] for d in docs]
#         emb_texts = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
#         q_emb = self.embedder.encode([query], convert_to_numpy=True)[0]
        
#         # 4. Rank
#         def cosine(a, b): 
#             return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
            
#         sims = [cosine(e, q_emb) for e in emb_texts]
#         top_idx = np.argsort(sims)[::-1][:TOP_PASSAGES]
#         top_passages = [{"url": docs[i]["url"], "passage": docs[i]["passage"], "score": float(sims[i])} for i in top_idx]

#         # 5. Summarize (Extractive) with Numbered Citations
#         sentences = []
#         for tp in top_passages:
#             for s in split_sentences(tp["passage"]):
#                 sentences.append({"sent": s, "url": tp["url"]})
        
#         references = []      # List to store unique URLs in order of appearance
#         url_to_id = {}       # Map URL string to numeric ID (1, 2, 3...)
        
#         if not sentences:
#             summary = "No summary could be generated."
#         else:
#             sent_texts = [s["sent"] for s in sentences]
#             sent_embs = self.embedder.encode(sent_texts, convert_to_numpy=True, show_progress_bar=False)
#             sent_sims = [cosine(e, q_emb) for e in sent_embs]
            
#             top_sent_idx = np.argsort(sent_sims)[::-1][:SUMMARY_SENTENCES]
#             chosen = [sentences[idx] for idx in top_sent_idx]

#             # De-duplicate and format
#             seen_text = set()
#             lines = []
            
#             for s in chosen:
#                 # Deduplication logic
#                 key = s["sent"].lower()[:80] 
#                 if key in seen_text:
#                     continue
#                 seen_text.add(key)
                
#                 # Citation Logic
#                 url = s['url']
#                 if url not in url_to_id:
#                     # New source found
#                     url_to_id[url] = len(references) + 1
#                     references.append(url)
                
#                 ref_id = url_to_id[url]
#                 lines.append(f"{s['sent']} [{ref_id}]")
            
#             summary = " ".join(lines)

#         elapsed = time.time() - start
#         return {
#             "query": query, 
#             "passages": top_passages, 
#             "summary": summary, 
#             "references": references, # Return the ordered list of sources
#             "time": elapsed
#         }