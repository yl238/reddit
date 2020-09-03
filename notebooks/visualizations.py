import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
from sklearn.metrics import confusion_matrix

def plot_tree(clf, feature_names=None, class_names=None):
    """Visualize the tree"""
    import graphviz
    from sklearn import tree
    dot_data = tree.export_graphviz(clf, out_file=None, 
                          feature_names=feature_names,  
                          class_names=class_names,  
                          filled=True, rounded=True,  
                                    special_characters=True)  
    graph = graphviz.Source(dot_data)
    graph.render('figures/decision_tree.png')
    
    
def evaluate(clf, plot=True):
    """Evaluate test set performance"""
    y_pred = clf.predict(X_test)    
    if plot:
        # Compute ane plot confusion matrix as heatmap
        cf = confusion_matrix(y_test, y_pred)
        df_cf = pd.DataFrame(cf, columns=clf.classes_, index=clf.classes_)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_cf, ax=ax, annot=True, cmap='Blues')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')