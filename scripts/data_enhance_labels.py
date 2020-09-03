import pandas as pd

if __name__ == '__main__':
    # Combine agreed on majority labels with SL's labels.
    TARGET = 'label'
    COLUMNS = ['title', 'score', 'num_comments', 'created_at', 'url', 'body']

    raw_file = '../datasets/reddit_data_more_labels.csv'
    raw_df = pd.read_csv(raw_file)

    df = raw_df[COLUMNS + ['category_sl']].iloc[:499, :]
    df['category_sl'] = df['category_sl'].str.lower()

    cleaned_file = '../datasets/first_100_labelled.csv'
    clean_df = pd.read_csv(cleaned_file)
    clean_df = clean_df[COLUMNS + ['majority_vote']]

    merged_df = pd.merge(df, clean_df[['title', 'majority_vote']], on='title', how='left')
    merged_df['label'] = merged_df['category_sl']
    merged_df['label'].iloc[:102] = merged_df['majority_vote'].iloc[:102]

    merged_df.to_csv('../datasets/reddit_raw_with_labels.csv', index=False)

    df = pd.read_csv('../datasets/reddit_raw_with_labels.csv')
    df2 = pd.read_csv('../datasets/labels_700-998.csv')
    
    
    df_concat = pd.concat([df[COLUMNS+[TARGET]], df2[COLUMNS+[TARGET]]])
    df_concat.to_csv('../datasets/all_reddit_labelled.csv', index=False)

    df = df_concat.copy()
    # Subsample the 'other' category
    LABELS = ['other company',
            'screeners', 
            'bad test', 
            'ratings', 
            'recorder', 
            'live convo', 
            'no test', 
            'mobile', 
            'bug', 
            'payment']
    df_other = df[df[TARGET] == 'other']
    df_other_sampled = df_other.sample(frac=0.5)
    df_other_labels = df[df[TARGET].isin(LABELS)]
    df_resampled = pd.concat([df_other_labels, df_other_sampled]).sort_values(by='created_at').reset_index()
    df_resampled.to_csv('../datasets/final_reddit_training.csv', index=False)