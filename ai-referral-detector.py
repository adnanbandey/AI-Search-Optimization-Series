import pandas as pd
import re

df = pd.read_csv("ga4_traffic.csv")

# Regex pattern for AI referrers
pattern = re.compile(r"(?i).*(openai\.com|chatgpt|you\.com|copilot\.microsoft\.com|gemini|perplexity|claude|anthropic|deepseek|huggingface\.co|llama|mistral|ai-bot|ai-crawler|web-llm).*")

# Flag AI-driven sessions
df['is_ai_traffic'] = df['page_referrer'].apply(lambda x: bool(pattern.match(str(x))))

# Distribution across common AI sources
ai_sources = ["chatgpt", "perplexity", "anthropic", "claude", "bingcopilot", "gemini"]
ai_counts = df[df['is_ai_traffic']].page_referrer.value_counts()

print("AI-driven sessions:", df['is_ai_traffic'].sum())
print("Top AI sources:\n", ai_counts.head())
