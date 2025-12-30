import os
import time
import google.generativeai as genai
import chromadb
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.text import Text
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
# PASTE YOUR KEY HERE IF .ENV FAILS
API_KEY = os.getenv("GOOGLE_API_KEY", "")

# --- SETUP DASHBOARD ---
console = Console()
layout = Layout()
layout.split_column(
    Layout(name="header", size=3),
    Layout(name="main", ratio=1),
    Layout(name="footer", size=3)
)

# --- SETUP AI & DB ---
genai.configure(api_key="")
# We use the safest, most common model name
model = genai.GenerativeModel('gemini-1.5-flash') 

# Setup Local Vector DB (The Brain)
client = chromadb.Client()
try:
    # Delete old collections to start fresh
    client.delete_collection("semantic_cache")
except:
    pass
cache_collection = client.create_collection("semantic_cache")

# --- FUNCTIONS ---

def get_embedding(text):
    """Turns text into numbers (vectors) using Gemini"""
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document",
    )
    return result['embedding']

def generate_answer(query):
    """Asks the AI for an answer (Slow/Expensive way)"""
    response = model.generate_content(query)
    return response.text

def smart_query(query):
    """The Optimized Path: Check Cache -> If Fail -> Ask AI"""
    start = time.time()
    
    # 1. Turn Question into Vector
    vector = get_embedding(query)
    
    # 2. Search Cache
    results = cache_collection.query(
        query_embeddings=[vector],
        n_results=1
    )
    
    # 3. Decision Logic
    is_hit = False
    answer = ""
    
    # Check if we found something close (Distance < 0.3 means similar)
    if results['distances'][0] and results['distances'][0][0] < 0.3:
        is_hit = True
        answer = results['documents'][0][0]
        source = "[bold green]⚡ CACHE HIT (Memory)[/]"
    else:
        # MISS: Ask AI
        answer = generate_answer(query)
        source = "[bold red]🐢 CACHE MISS (AI Call)[/]"
        # Save to memory for next time
        cache_collection.add(
            documents=[answer],
            embeddings=[vector],
            ids=[str(time.time())]
        )

    end = time.time()
    duration = end - start
    return answer, duration, source, is_hit

# --- THE UI LOOP ---
def run_dashboard():
    history = []
    
    header_text = Text("🚀 SEMANTIC CACHE OPTIMIZER", style="bold white on blue", justify="center")
    layout["header"].update(Panel(header_text))

    with Live(layout, refresh_per_second=4) as live:
        while True:
            # Update Table
            table = Table(title="Live Request Log")
            table.add_column("Query", style="cyan")
            table.add_column("Source", justify="center")
            table.add_column("Latency", justify="right")
            table.add_column("Speedup", justify="right", style="bold yellow")

            baseline = 1.5 # Assume 1.5s is normal AI speed
            
            for h in history[-8:]: # Show last 8
                speedup = f"{baseline / h['time']:.1f}x" if h['hit'] else "1.0x"
                table.add_row(h['query'], h['source'], f"{h['time']:.4f}s", speedup)
            
            layout["main"].update(Panel(table))
            
            # Input Prompt
            layout["footer"].update(Panel("Type a question (or 'exit'): ", style="bold green"))
            live.refresh()
            
            # Get User Input
            query = console.input("[bold green]Enter Question > [/]")
            if query.lower() in ['exit', 'quit']:
                break
                
            # Run System
            layout["footer"].update(Panel(f"🧠 Processing: {query}...", style="blink yellow"))
            live.refresh()
            
            ans, duration, source, is_hit = smart_query(query)
            
            # Add to history
            history.append({
                "query": query,
                "source": source,
                "time": duration,
                "hit": is_hit
            })
            
            # Show Answer briefly
            console.print(Panel(f"[bold white]{ans}[/]", title="Answer", border_style="green"))
            time.sleep(2) # Pause so user can read

if __name__ == "__main__":
    run_dashboard()
