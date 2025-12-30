# ragtool

* **Semantic Understanding:** Detects similar questions even if phrased differently (e.g., "Hello" vs "Hi there").
* **Vector Database:** Uses **ChromaDB** locally to store embeddings of questions and answers.
* **Google Gemini Powered:** Uses `gemini-1.5-flash` for generation and `text-embedding-004` for vectorization.
* **Live Dashboard:** Real-time visualization of latency, speedup factors, and request history.
* **Cost Efficiency:** Prevents redundant API calls for repeated questions.
