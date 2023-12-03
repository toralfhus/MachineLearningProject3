import numpy as np

from utils import *
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score
from xgboost import XGBClassifier

PATH_RADIOMICS = "radiomics_fbc=32_train.csv"
df = pd.read_csv(PATH_RADIOMICS, index_col=0)
print(df["shape"].unique())
print(df["PixelSpacing"].unique())
X = df.drop(columns=["shape", "PixelSpacing"]).dropna()
print(X.shape)


y = load_outcome(df.index.values)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_test.shape)

rf = RandomForestClassifier()
lr = LogisticRegression(max_iter=int(1e3), penalty="elasticnet", solver="saga", l1_ratio=0.9, C=5.0)
xgb = XGBClassifier()


for md, m in {"RF":rf, "LR":lr, "XGB":xgb}.items():


    m.fit(X_train, y_train)
    yhat_train = m.predict(X_train)
    yhat = m.predict(X_test)


    f1 = f1_score(y_test, yhat)
    acc = accuracy_score(y_test, yhat)
    auc = roc_auc_score(y_test, yhat)
    rec = recall_score(y_test, yhat)
    print(md, "Acc / AUC / Recall / F1 = ", np.round([acc, auc, rec, f1], 3), end="\t")

    f1 = f1_score(y_train, yhat_train)
    acc = accuracy_score(y_train, yhat_train)
    auc = roc_auc_score(y_train, yhat_train)
    rec = recall_score(y_train, yhat_train)
    print(np.round([acc, auc, rec, f1], 3))


    if md == "RF":
        # ft_importances = {ft:imp for ft, imp in zip(m.feature_names_in_, m.feature_importances_)}
        # ft_importances_sorted = dict(sorted(ft_importances.items(), key=lambda x:x[1], reverse=True))
        print("\tNonzero:", np.count_nonzero(m.feature_importances_))
    #     for i in range(10):
    #         print(list(ft_importances_sorted.items())[i])
    #     plt.plot(range(len(ft_importances)), ft_importances.values(), "x")
    #     plt.show()
    if md == "LR":
        print("\tNonzero:", np.count_nonzero(m.coef_))
