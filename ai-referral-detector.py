import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("ga_llm.csv",skiprows=7)

# Regex pattern for AI referrers
# pattern = re.compile(r"(?i).*(openai\.com|chatgpt|you\.com|copilot\.microsoft\.com|gemini|perplexity|claude|anthropic|deepseek|huggingface\.co|llama|mistral|ai-bot|ai-crawler|web-llm).*")

df1 = df.drop(columns=['Grand total'])

df1.columns = [
    'page_referrer',   # formerly 'Unnamed: 0'
    'landing_page',       # formerly 'Unnamed: 1'
    'session',    
]


df1['landing_page'] = df1['landing_page'].astype(str).str.strip()
df1['page_referrer'] = df1['page_referrer'].astype(str).str.strip()

df1.columns

df_referrer = (
    df1.groupby('page_referrer', as_index=False)['session']
       .sum()
       .sort_values(by='session', ascending=False)
       .reset_index(drop=True)
)

df_referrer.head(5)

# Step 1: Define mapping keywords → LLM names
llm_map = {
    "chatgpt": "ChatGPT",
    "perplexity": "Perplexity",
    "anthropic": "Anthropic",
    "claude": "Claude",
    "bingcopilot": "Bing Copilot",
    "copilot": "Bing Copilot",   # if you want copilot to map to same
    "gemini": "Gemini"
}

# Step 2: Create new column 'llm' based on pattern matching
df_referrer['llm'] = df_referrer['page_referrer'].str.lower().apply(
    lambda x: next((v for k, v in llm_map.items() if k in x), np.nan)
)

# Step 3: Filter rows where classification succeeded
df_llm = df_referrer.dropna(subset=['llm'])

# Step 4: Sum sessions by LLM
df_llm_grouped = (
    df_llm.groupby('llm', as_index=False)['session']
          .sum()
          .sort_values(by='session', ascending=False)
          .reset_index(drop=True)
)

# Step 5: Calculate % share
total = df_llm_grouped['session'].sum()
df_llm_grouped['percentage'] = (df_llm_grouped['session'] / total) * 100

plt.figure(figsize=(8, 5))

bars = plt.bar(df_llm_grouped['llm'], df_llm_grouped['percentage'])

# Add data labels above bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height,
             f'{height:.0f}',
             ha='center', va='bottom')

plt.xlabel("LLM Source")
plt.ylabel("Sessions")
plt.title("Sessions by LLM Referral Source")
plt.tight_layout()
plt.show()
