

import os, re, html, time
import pandas as  pd
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv("ecommerce_prompts.csv")

# df['Prompt'].unique().tolist()

# ---------------------------------------------------------
# STEP 1: CONFIGURATION
# ---------------------------------------------------------
filename = 'chatgpt.com.har'
print(f"Reading file: {filename}...")

with open(filename, 'r', encoding='utf-8') as f:
    har_data = json.load(f)

all_found_citations = []

# ---------------------------------------------------------
# STEP 2: LOOP THROUGH NETWORK TRAFFIC
# ---------------------------------------------------------
entries = har_data['log']['entries']
print(f"Total network requests found: {len(entries)}")

for i, entry in enumerate(entries):
    response = entry.get('response', {})
    content = response.get('content', {})
    text_body = content.get('text', '')

    if not text_body:
        continue
    
    # Reset the prompt for each new network request
    current_prompt = "Unknown Prompt"

    # ---------------------------------------------------------
    # STEP 3: PARSE THE STREAM (Line by Line)
    # ---------------------------------------------------------
    lines = text_body.split('\n')
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('data: '):
            json_str = line[6:] 
            
            if json_str in ('[DONE]', '"v1"'):
                continue

            try:
                json_data = json.loads(json_str)

                # ---------------------------------------------------------
                # STEP 3.5: CAPTURE THE PROMPT (New Logic)
                # ---------------------------------------------------------
                # We check if this line is the "input_message" you found
                if json_data.get('type') == 'input_message':
                    try:
                        # Extract the text from parts: ["What do users say...?"]
                        current_prompt = json_data['input_message']['content']['parts'][0]
                    except (KeyError, IndexError, TypeError):
                        pass # Keep default if extraction fails
                
                # ---------------------------------------------------------
                # STEP 4: SEARCH INSIDE FRAGMENT
                # ---------------------------------------------------------
                stack = [json_data]

                while stack:
                    current_item = stack.pop()

                    if isinstance(current_item, dict):
                        found_source = current_item.get('attribution') or current_item.get('domain')
                        is_search_result = 'search_result' in str(current_item.get('type', ''))
                        
                        if found_source and (is_search_result or 'pub_date' in current_item):
                            
                            # Get ref_type/index from the nested 'ref_id' dict if it exists
                            ref_data = current_item.get('ref_id', {})
                            ref_type = ref_data.get('ref_type') or current_item.get('ref_type', 'N/A')
                            ref_index = ref_data.get('ref_index') or current_item.get('ref_index', 'N/A')

                            all_found_citations.append({
                                'prompt': current_prompt,  # <--- Added this column
                                'source': found_source,
                                'pub_date': current_item.get('pub_date'),
                                'title': current_item.get('title', 'N/A'),
                                'url': current_item.get('url', 'N/A'),
                                'ref_type': ref_type,
                                'ref_index': ref_index
                            })

                        for key, value in current_item.items():
                            if isinstance(value, (dict, list)):
                                stack.append(value)

                    elif isinstance(current_item, list):
                        for item in current_item:
                            if isinstance(item, (dict, list)):
                                stack.append(item)

            except json.JSONDecodeError:
                continue

# ---------------------------------------------------------
# STEP 5: SAVE RESULTS
# ---------------------------------------------------------
print(f"Total citations found: {len(all_found_citations)}")

fd = pd.DataFrame(all_found_citations)

fd1 = fd[ ( ~fd['prompt'].isna() ) & (fd['ref_type']!= "N/A") ].reset_index(drop=True)

fd1

df.head()

fd1['prompt_clean'] = fd1['prompt'].astype(str).str.strip().str.lower()
df['prompt_clean'] = df['Prompt'].astype(str).str.strip().str.lower()

fd1['prompt_clean'] = fd1['prompt'].str.replace("'","")
df['prompt_clean'] = df['Prompt'].str.replace("'","")

# 2. MERGE: Left join to keep all your scraped results
# We merge on the cleaned columns but keep the original 'Prompt' for display
final_df = pd.merge(
    fd1[['source', 'pub_date', 'title', 'url', 'ref_type', 'ref_index','prompt_clean']], 
    df[['Category', 'prompt_clean']], # Only bring over Category and the join key
    on='prompt_clean', 
    how='left'
)

# 3. CLEANUP: Drop the temporary helper column
final_df = final_df.rename(columns={'prompt_clean':'Prompt'})

fd1.columns

final_df.columns

final_df['pub_date_readable'] = pd.to_datetime(final_df['pub_date'], unit='s', errors='coerce')

final_df['pub_date_readable']

# =========================================================
# 1. DATA PREPARATION & CLEANING
# =========================================================
# Assume 'final_df' is your merged dataframe from the previous step.
# Create a copy to avoid SettingWithCopy warnings
df_viz = final_df.copy()

# CRITICAL FIX: Force ref_index to numeric, turning 'N/A' into NaN
df_viz['ref_index'] = pd.to_numeric(df_viz['ref_index'], errors='coerce')

# Fix Date format
df_viz['pub_date'] = pd.to_numeric(df_viz['pub_date'], errors='coerce')
df_viz['pub_date_obj'] = pd.to_datetime(df_viz['pub_date'], unit='s', errors='coerce')
df_viz['article_age_days'] = (pd.Timestamp.now() - df_viz['pub_date_obj']).dt.days

# Drop rows where Category is missing for cleaner plots
df_viz = df_viz.dropna(subset=['Category'])

# DEFINING COLORS
# Using a distinct color palette
distinct_colors = px.colors.qualitative.Dark24  # A palette with 24 distinct colors

# =========================================================
# 2. PLOT CONFIGURATION (Height/Width Control)
# =========================================================
# Change these values to resize your charts easily
H_PLOT1 = 600
W_PLOT1 = 1000

H_PLOT2 = 900  # Made taller as requested
W_PLOT2 = 1000

H_PLOT3 = 800  # Made bigger
W_PLOT3 = 1200

H_PLOT4 = 700
W_PLOT4 = 1000

H_PLOT5 = 600
W_PLOT5 = 1000

# Helper function to remove gridlines and add labels
def clean_layout(fig):
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False, visible=True),
        yaxis=dict(showgrid=False, showticklabels=False, title=None), # Hide Y axis ticks/labels
        margin=dict(l=20, r=20, t=50, b=50)
    )
    return fig

# =========================================================
# VIZ 1: Category vs Ref Type (Counts)
# =========================================================
# Data Prep
viz1_data = df_viz.groupby(['Category', 'ref_type']).size().reset_index(name='count')

fig1 = px.bar(
    viz1_data, 
    x='Category', 
    y='count', 
    color='ref_type',
    title='<b>Count of References by Category (Split by Type)</b>',
    text='count',
    height=H_PLOT1,
    width=W_PLOT1,
    color_discrete_sequence=px.colors.qualitative.Safe # Distinct colors for types
)
fig1.update_traces(textposition='auto')
fig1 = clean_layout(fig1)
fig1.show()

# =========================================================
# VIZ 1 (PERCENTAGE): Category vs Ref Type Distribution (%)
# =========================================================

# 1. Calculate raw counts
viz1_data = df_viz.groupby(['Category', 'ref_type']).size().reset_index(name='count')

# 2. Calculate total counts per category
cat_totals = viz1_data.groupby('Category')['count'].transform('sum')

# 3. Calculate percentage
viz1_data['percentage'] = (viz1_data['count'] / cat_totals) * 100

# 4. Format label (e.g., "45.2%")
viz1_data['label'] = viz1_data['percentage'].round(1).astype(str) + '%'

# 5. Plot
fig1_pct = px.bar(
    viz1_data, 
    x='Category', 
    y='percentage', 
    color='ref_type',
    title='<b>Reference Type Distribution by Category (%)</b>',
    text='label',
    height=H_PLOT1,
    width=W_PLOT1,
    color_discrete_sequence=px.colors.qualitative.Safe
)

fig1_pct.update_traces(textposition='inside')
fig1_pct.update_layout(
    plot_bgcolor='white',
    barmode='stack', # Stack bars to show 100% total
    xaxis=dict(showgrid=False, title=None),
    yaxis=dict(showgrid=False, showticklabels=False, title=None, range=[0, 100]), # Hide Y axis, fix range to 100
    margin=dict(l=20, r=20, t=50, b=50)
)
fig1_pct.show()

# =========================================================
# VIZ 2: Category vs Recency (Boxplot)
# =========================================================
# Lower value = Newer article
fig2 = px.box(
    df_viz, 
    x='Category', 
    y='article_age_days',
    color='Category',
    title='<b>Article Freshness by Category (Lower is Newer)</b>',
    height=H_PLOT2,
    width=W_PLOT2,
    color_discrete_sequence=distinct_colors
)
# We can't easily put text labels on every dot of a boxplot, 
# but hovering will show details. We keep the axis for context here.
fig2.update_layout(
    plot_bgcolor='white',
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, title='Age in Days'), # Keep grid here for readability of scale
    showlegend=False
)
fig2.show()

# =========================================================
# VIZ 3: Top 3 Sources per Category
# =========================================================
# Data Prep
top_sources = (
    df_viz.groupby(['Category', 'source'])
    .size()
    .reset_index(name='count')
    .sort_values(['Category', 'count'], ascending=[True, False])
    .groupby('Category')
    .head(3)
)

fig3 = px.bar(
    top_sources, 
    x='source', 
    y='count', 
    color='Category', # Coloring by category
    facet_col='Category', 
    facet_col_wrap=3, # How many charts per row
    title='<b>Top 3 Sources per Category</b>',
    text='count',
    height=H_PLOT3,
    width=W_PLOT3,
    color_discrete_sequence=distinct_colors
)
# Clean up the facet labels
fig3.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig3.update_xaxes(matches=None, showticklabels=True) # Allow different sources on X axis
fig3.update_yaxes(matches=None, showticklabels=False, showgrid=False)
fig3.update_traces(textposition='outside')
fig3.show()

# =========================================================
# VIZ 4: Top 15 Sources Overall (Horizontal)
# =========================================================
# Data Prep
top_15_sources = df_viz['source'].value_counts().head(15).reset_index()
top_15_sources.columns = ['source', 'count']
top_15_sources = top_15_sources.sort_values('count', ascending=True) # Sort for horiz bar

fig4 = px.bar(
    top_15_sources, 
    x='count', 
    y='source', 
    orientation='h',
    title='<b>Top 15 Most Cited Sources (Overall)</b>',
    text='count',
    height=H_PLOT4,
    width=W_PLOT4,
    color='count', # Gradient color based on count
    color_continuous_scale='Viridis'
)
fig4.update_traces(textposition='outside')
fig4.update_layout(
    plot_bgcolor='white',
    xaxis=dict(showgrid=False, showticklabels=False), # Hide X numbers
    yaxis=dict(showgrid=False) # Keep Source names
)
fig4.show()

# =========================================================
# VIZ 5: Search Depth (Avg Max Ref Index)
# =========================================================
# Data Prep
prompt_depth = df_viz.groupby(['Category', 'Prompt'])['ref_index'].max().reset_index()
category_depth = prompt_depth.groupby('Category')['ref_index'].mean().reset_index()
category_depth['ref_index'] = category_depth['ref_index'].round(1) # Clean decimals
category_depth = category_depth.sort_values('ref_index', ascending=False)

fig5 = px.bar(
    category_depth, 
    x='Category', 
    y='ref_index', 
    color='Category',
    title='<b>Average Search Depth (How deep did the AI look?)</b>',
    text='ref_index',
    height=H_PLOT5,
    width=W_PLOT5,
    color_discrete_sequence=distinct_colors
)
fig5.update_traces(textposition='outside')
fig5 = clean_layout(fig5)
fig5.show()



