import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the CSV files
df_30 = pd.read_csv('./pos_tokens/pos_removal_statistics_30_imdb.csv')
df_50 = pd.read_csv('./pos_tokens/pos_removal_statistics_50_imdb.csv')
df_70 = pd.read_csv('./pos_tokens/pos_removal_statistics_70_imdb.csv')
df_90 = pd.read_csv('./pos_tokens/pos_removal_statistics_90_imdb.csv')

# Add a column to each DataFrame to indicate the model
df_30['model'] = '30%'
df_50['model'] = '50%'
df_70['model'] = '70%'
df_90['model'] = '90%'

# Combine all DataFrames
combined_df = pd.concat([df_30, df_50, df_70, df_90])

# Drop the unwanted POS tags
combined_df = combined_df[~combined_df['pos'].isin(['X', 'SYM', 'INTJ'])]

# Define aggregation mapping
aggregation_mapping = {
    'NOUN': 'NOUN',
    'PROPN': 'NOUN',
    'VERB': 'VERB',
    'AUX': 'VERB',
    'ADJ': 'ADJ_ADV',
    'ADV': 'ADJ_ADV',
    'CCONJ': 'CONJ',
    'SCONJ': 'CONJ',
    'DET': 'DET_PRON',
    'PRON': 'DET_PRON',
    'PUNCT': 'PUNCT',
    'ADP': 'OTHER',
    'NUM': 'OTHER',
    'PART': 'OTHER',
}

# Apply aggregation mapping
combined_df['aggregated_pos'] = combined_df['pos'].map(aggregation_mapping)

# Ensure no relative removal rate exceeds 1
combined_df['relative_removal'] = combined_df['relative_removal'].apply(lambda x: min(x, 1))

# Pivot the DataFrame to get aggregated POS tags as rows and models as columns
pivot_df = combined_df.pivot_table(index='aggregated_pos', columns='model', values='relative_removal', aggfunc='mean')

# Reorder the columns to be in the order 90%, 70%, 50%, 30%
pivot_df = pivot_df[['90%', '70%', '50%', '30%']]

# Create an interactive line plot using Plotly
fig = go.Figure()

for pos_tag in pivot_df.index:
    fig.add_trace(go.Scatter(
        x=pivot_df.columns,
        y=pivot_df.loc[pos_tag],
        mode='lines+markers',
        name=pos_tag
    ))

fig.update_layout(
    title='POS Token Relative Removal Rates across Different Models',
    xaxis_title='Model',
    yaxis_title='Relative Removal Rate',
    legend_title='Aggregated Part of Speech',
    template='plotly_dark'
)

fig.show()