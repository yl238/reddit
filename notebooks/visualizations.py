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