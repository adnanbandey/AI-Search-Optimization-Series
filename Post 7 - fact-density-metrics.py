import re
import spacy
import pandas as pd
import trafilatura
from playwright.async_api import async_playwright # <--- CHANGED TO ASYNC
import os
import shutil

# --- FIX: OVERRIDE SYSTEM TEMP DIRECTORY ---
# This forces Playwright to create temp files in your local project folder
# instead of the blocked /var/folders/... system path.
local_temp = os.path.abspath("./temp_browser_data")
os.makedirs(local_temp, exist_ok=True)
os.environ["TMPDIR"] = local_temp  # Tell Playwright to use this!

# --- OPTIONAL: CLEANUP OLD RUNS ---
# If you want to start fresh every time, uncomment the next line:
# shutil.rmtree(local_temp, ignore_errors=True); os.makedirs(local_temp, exist_ok=True)

print(f">>> Overriding temp dir to: {local_temp}")

# --- YOUR EXISTING IMPORTS ---
import re
import spacy
import pandas as pd
import trafilatura
from playwright.async_api import async_playwright

# --- 1. CONFIGURATION ---
URLS_TO_TEST = [
    "https://www.toyota.com.au/bz4x-ev",
    "https://www.toyota.com.au/hilux"
]

# --- 2. SETUP NLP & REGEX ---
print(">>> Loading Spacy model...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: Spacy model not found. Run: !python -m spacy download en_core_web_sm")
    # In Jupyter, you might need to restart the kernel after installing
    raise

NUM_PATTERN = re.compile(r"""
    (\b\d{1,3}(?:[,\s]\d{3})+\b)   # 12,345
  | (\b\d+(?:\.\d+)?%?\b)          # 12.5%
  | (\b[A-Z]{1,4}\d{1,4}\b)        # Model codes
""", re.X)

results_data = []
top_sentences_log = {}

# --- 3. START BROWSER (ASYNC) ---
print(">>> Launching Headless Browser...")

# In Jupyter, we use 'await' directly:
p = await async_playwright().start() 
browser = await p.chromium.launch(headless=True)
context = await browser.new_context(
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
)
page = await context.new_page()

# --- 4. MAIN LOOP ---
try:
    for url in URLS_TO_TEST:
        print(f"\n--- Processing: {url} ---")
        
        # A. FETCH HTML (Async wait)
        print("   Fetching page...")
        try:
            # We await the network idle state
            await page.goto(url, wait_until="networkidle", timeout=30000)
            html_content = await page.content()
        except Exception as e:
            print(f"   !! Error fetching {url}: {e}")
            continue

        from bs4 import BeautifulSoup

        print("   Extracting text (BeautifulSoup method)...")
        
        # 1. Parse the HTML
        soup = BeautifulSoup(html_content, "html.parser")
        
        # 2. Remove "Junk" elements that confuse the score
        # We strip scripts, styles, SVGs (icons), and navigation/footers if you want
        for element in soup(["script", "style", "svg", "noscript", "meta"]):
            element.decompose() # Completely remove these from the tree
        
        # 3. specific cleanup for "hidden" text (optional but recommended)
        # Many sites hide text for mobile/desktop versions; we generally want visible text.
        # This is hard to do perfectly without the browser, but we can guess.
        
        # 4. Extract Text with Separators
        # 'separator="\n"' ensures headings don't merge into the next sentence.
        raw_text = soup.get_text(separator="\n", strip=True)
        
        # 5. Post-Processing
        # Marketing pages have lots of empty newlines. Let's clean that up.
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        
        # Rejoin into a clean block of text
        cleaned_text = "\n".join(lines)
        
        # C. ANALYZE (Standard Python logic)
        doc = nlp(cleaned_text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        total_sentences = len(sents)
        total_words = sum(len(s.split()) for s in sents)
        
        if total_sentences == 0:
            continue

        fact_flags = []
        scored_sentences = []
        
        for s in sents:
            ents = [e for e in nlp(s).ents]
            has_num = bool(NUM_PATTERN.search(s))
            
            is_fact = (len(ents) > 0) or has_num
            fact_flags.append(is_fact)
            
            score = len(ents) + 0.8 * len(NUM_PATTERN.findall(s))
            if score > 0:
                scored_sentences.append((score, s))

        fact_density = sum(fact_flags) / total_sentences
        facts_per_100w = 100.0 * sum(fact_flags) / max(1, total_words)

        # Lead Density
        acc_words = 0
        lead_flags = []
        for i, s in enumerate(sents):
            acc_words += len(s.split())
            lead_flags.append(fact_flags[i])
            if acc_words >= 500: break
        
        lead_density = (sum(lead_flags) / len(lead_flags)) if lead_flags else 0.0

        results_data.append({
            "url": url,
            "sentences": total_sentences,
            "words": total_words,
            "fact_density": round(fact_density, 3),
            "facts_per_100w": round(facts_per_100w, 1),
            "lead500_density": round(lead_density, 3)
        })

        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        top_sentences_log[url] = [s for score, s in scored_sentences[:5]]
        
        print(f"   Done. Density: {fact_density:.2f}")

finally:
    print("\n>>> Closing Browser...")
    await browser.close()
    await p.stop()

# --- 5. OUTPUT ---
print("\n" + "="*40)
print("RESULTS")
print(pd.DataFrame(results_data).to_string(index=False))

print("\nTOP SENTENCES")
for url, sentences in top_sentences_log.items():
    print(f"\nSOURCE: {url}")
    for i, s in enumerate(sentences, 1):
        print(f"{i}. {s}")
