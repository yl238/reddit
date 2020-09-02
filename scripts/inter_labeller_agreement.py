import pandas as pd
import itertools
from sklearn.metrics import cohen_kappa_score

LABEL_COLS = ['category_jj', 'category_st', 'category_sl']

def get_agreement(df, labeller1, labeller2):
    score = cohen_kappa_score(df[labeller1], df[labeller2])
    output = "kappa({}, {}) = {}".format(labeller1, labeller2, score)
    return output

if __name__ == '__main__':
    df = pd.read_csv('../datasets/first_100_labelled.csv')
    LABELLERS = LABEL_COLS + ['majority_vote']
    for i, j in itertools.combinations(LABELLERS, 2):
        print(get_agreement(df, i, j))