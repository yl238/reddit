from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def eval_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    return {'accuracy': accuracy, 'f1_score': f1, 'precision': precision, 'recall': recall}