import pandas as pd

if __name__ == '__main__':
    TARGET = 'label'
    COLUMNS = ['title', 'score', 'num_comments', 'created_at', 'url', 'body']

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

    df = pd.read_csv('../datasets/all_reddit_labelled.csv')
    df_other = df[df[TARGET] == 'other']
    
    # Subsample the other category by 50%
    df_other_sampled = df_other.sample(frac=0.5, random_state=42)

    df_other_labels = df[df[TARGET].isin(LABELS)]
    df_resampled = pd.concat([df_other_labels, df_other_sampled]).sort_values(by='created_at').reset_index()
    df_resampled.to_csv('../datasets/final_reddit_training.csv', index=False)