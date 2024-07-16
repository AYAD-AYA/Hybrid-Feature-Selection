import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE

pd.set_option("display.max_columns", None)


def proposed_fs_model(file, target_column, n_features_to_select):
    filen = file.copy()

    # Correlation matrix with Pearson
    corr_matrix1 = filen.corr(method='pearson')
    upper1 = corr_matrix1.where(np.triu(np.ones(corr_matrix1.shape), k=1).astype(bool))
    to_drop1 = [column for column in upper1.columns if any(upper1[column] > 0.6)]
    file1 = filen.drop(columns=to_drop1)

    # Correlation matrix with Spearman
    corr_matrix2 = filen.corr(method='spearman')
    upper2 = corr_matrix2.where(np.triu(np.ones(corr_matrix2.shape), k=1).astype(bool))
    to_drop2 = [column for column in upper2.columns if any(upper2[column] > 0.6)]
    file2 = filen.drop(columns=to_drop2)

    list_pcc = file1.columns.tolist()
    list_scc = file2.columns.tolist()
    print(list_pcc)
    print(list_scc)
    encoder = LabelEncoder()

    for each in file.columns:
        file[each] = encoder.fit_transform(filen[each])

    x = file.drop(columns=target_column)
    y = file[target_column]

    model1 = DecisionTreeClassifier()
    rfe1 = RFE(model1, n_features_to_select=n_features_to_select)
    rfe1.fit(x, y)

    model2 = RandomForestClassifier(n_estimators=10)
    rfe2 = RFE(model2, n_features_to_select=n_features_to_select)
    rfe2.fit(x, y)

    list_rf = rfe1.support_
    list_dt = rfe2.support_

    features = filen.drop(columns=target_column).columns.tolist()

    features_rf = [features[i] for i in range(len(list_rf)) if list_rf[i]]
    features_dt = [features[i] for i in range(len(list_dt)) if list_dt[i]]
    print(features_dt)
    print(features_rf)
    # Intersection
    S1 = list(set(list_pcc) & set(list_scc))
    S2 = list(set(features_dt) & set(features_rf))
    print(S1)
    print(S2)

    # Union
    S = list(set(S1) | set(S2))

    print("last feature set:", S)
    print(len(S))

    return S
