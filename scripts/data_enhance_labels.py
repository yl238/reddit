import pandas as pd

# Combine agreed on majority labels with SL's labels.
VALID_COLUMNS = ['title', 'score', 'num_comments', 'created_at', 'body']

raw_file = '../datasets/reddit_data_more_labels.csv'
raw_df = pd.read_csv(raw_file)

df = raw_df[VALID_COLUMNS + ['category_sl']].iloc[:499, :]
df['category_sl'] = df['category_sl'].str.lower()

cleaned_file = '../datasets/first_100_labelled.csv'
clean_df = pd.read_csv(cleaned_file)
clean_df = clean_df[VALID_COLUMNS + ['majority_vote']]

merged_df = pd.merge(df, clean_df[['title', 'majority_vote']], on='title', how='left')

merged_df['label'] = merged_df['category_sl']
merged_df['label'].iloc[:102] = merged_df['majority_vote'].iloc[:102]

merged_df.to_csv('../datasets/reddit_raw_with_labels.csv', index=False)