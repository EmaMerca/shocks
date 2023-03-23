if __name__ == "__main__":

    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import json
    from collections import defaultdict
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import spearmanr
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    # only used for debugging, set it them to (1000, 10) for a fast run
    DATA_LIMIT = -1
    FEATURES_LIMIT = -1


    ### prepare featurized data ###
    ## ------------------------- ##

    with open("/Users/emamerca/dev/shocks/data/featurized/features__KO__2021-11-01_2022-01-01__30s__2_std.json", "r") as f:
        data = json.loads(f.read())

    data = [el for el in (data[0][:DATA_LIMIT] + data[1][:DATA_LIMIT])]
    # binarize predictions features (either shock or non shock).  We don't care about shock direction
    for el in data:
        direction = el['direction']
        el['direction'] = 1 if direction == -1 else direction

    df = pd.DataFrame.from_dict(data).dropna()
    cols = [col for col in df.columns if col != 'direction'][:FEATURES_LIMIT]
    labels = df['direction'].tolist()
    df = df[cols]

    X_train, X_test, y_train, y_test = train_test_split(df[cols], labels, test_size=0.3, random_state=42, stratify=labels)

    class_weights = {
        0: (len([el for el in labels if el == 0]) / len(labels)),
        1: (len([el for el in labels if el == 1]) / len(labels)),
    }

    x_train_idx_to_timestamp = {idx: ts for idx, ts in enumerate(X_train.time.values)}
    x_test_idx_to_timestamp = {idx: ts for idx, ts in enumerate(X_test.time.values)}

    X_train.drop("time", axis=1, inplace=True)
    X_test.drop("time", axis=1, inplace=True)

    clf = RandomForestClassifier(n_estimators=120, random_state=42, ccp_alpha=0, max_depth=25,
                                 criterion="gini", class_weight=class_weights)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)


    ### features selection ###
    ## -------------------- ##

    perm_importance = permutation_importance(clf, X_train, y_train, n_repeats=10, random_state=42)
    perm_sorted_idx = perm_importance.importances_mean.argsort()

    tree_importance_sorted_idx = np.argsort(clf.feature_importances_)
    tree_indices = np.arange(0, len(clf.feature_importances_)) + 0.5

    ## plot feature importances according to classifier
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    #ax1.barh(tree_indices, clf.feature_importances_[tree_importance_sorted_idx], height=0.7)
    #ax1.set_yticks(tree_indices)
    #ax1.set_yticklabels([col for idx, col in enumerate(X_train.columns.to_list()) if idx in tree_importance_sorted_idx])
    #ax1.set_ylim((0, len(clf.feature_importances_)))
    #ax2.boxplot(
    #    perm_importance.importances[perm_sorted_idx].T,
    #    vert=False,
    #    labels=[col for idx, col in enumerate(X_train.columns.to_list()) if idx in perm_sorted_idx],
    #)
    #fig.tight_layout()
    #plt.show()

    corr = spearmanr(X_train).correlation
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    dendro = hierarchy.dendrogram(
        dist_linkage, labels=X_train.columns.tolist(), ax=ax1, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro["ivl"]))

    # plot correlation matrix and dendrogram
    # ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
    # ax2.set_xticks(dendro_idx)
    # ax2.set_yticks(dendro_idx)
    # ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    # ax2.set_yticklabels(dendro["ivl"])
    # fig.tight_layout()
    # plt.show()


    # pick threshold and select features
    threshold = 1
    cluster_ids = hierarchy.fcluster(dist_linkage, threshold, criterion="distance")
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

    X_train_sel = X_train.iloc[:, selected_features]
    X_test_sel = X_test.iloc[:, selected_features]

    clf_sel = RandomForestClassifier(n_estimators=120, random_state=42, ccp_alpha=0, max_depth=25,
                                     criterion="gini", class_weight=class_weights)
    clf_sel.fit(X_train_sel, y_train)
    preds = clf_sel.predict(X_test_sel)
    predicted_shock_times = [x_test_idx_to_timestamp[idx] for idx, pred in enumerate(preds) if pred == 1]
