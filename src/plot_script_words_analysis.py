import pandas as pd

# Load the CSV file
csv_file_path = r'output\30_imdb_token_removal_statistics_gpt2.csv'  # Adjust the path as needed
combined_df = pd.read_csv(csv_file_path)

combined_df = combined_df[combined_df['token'] != '½']
# Set a minimum threshold for token appearances
min_appearances = 20

# Filter based on the minimum threshold
combined_df = combined_df[combined_df['token_appearance_count'] >= min_appearances]
# Display the combined DataFrame
print(combined_df.head())

# Plotting the top 20 tokens based on relative removal
import plotly.express as px

# Filter to the top 20 tokens based on relative removal
top_20_tokens = combined_df.nlargest(20, 'relative_removal')

# Display the filtered DataFrame
print(top_20_tokens)

# 1. Bar Plot: Showing the count of tokens removed for the top 20 tokens
bar_fig = px.bar(top_20_tokens, x='token', y='removed_token_count', title='Count of Tokens Removed (Top 20)', labels={'removed_token_count': 'Removed Token Count'})
bar_fig.show()

# 2. Scatter Plot: Showing the relationship between token appearance and token removal for the top 20 tokens
scatter_fig = px.scatter(top_20_tokens, x='token_appearance_count', y='removed_token_count', size='relative_removal', hover_name='token', title='Token Appearance vs. Removal (Top 20)')
scatter_fig.update_layout(xaxis_title='Token Appearance Count', yaxis_title='Removed Token Count')
scatter_fig.show()

# 3. Pie Chart: Showing the proportion of the top removed tokens
pie_fig = px.pie(top_20_tokens, names='token', values='removed_token_count', title='Proportion of Top 20 Removed Tokens')
pie_fig.show()

# Additional Plot Suggestions
# 4. Line Plot: Relative Removal over Tokens for the top 20 tokens
line_fig = px.line(top_20_tokens, x='token', y='relative_removal', title='Relative Removal Rate over Tokens (Top 20)', labels={'relative_removal': 'Relative Removal Rate'})
line_fig.show()

# 5. Histogram: Distribution of Token Appearance Counts for the top 20 tokens
hist_fig = px.histogram(top_20_tokens, x='token_appearance_count', nbins=20, title='Distribution of Token Appearance Counts (Top 20)', labels={'token_appearance_count': 'Token Appearance Count'})
hist_fig.show()