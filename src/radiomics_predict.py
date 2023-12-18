import numpy as np

from utils import *
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score, precision_score
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from joblib import dump, load


# PATH_RADIOMICS = "radiomics_fbc=32_train.csv"
# PATH_RADIOMICS = "radiomics_downsamp_fbc=32_train.csv"
PATH_RADIOMICS = "radiomics_seg_fbc=32_train.csv"


# PATH_TRAIN_TEST_SPLIT = "train_test_split.csv"
PATH_TRAIN_TEST_SPLIT = "train_test_split_segmented.csv"

# PATH_TRAIN_TEST_SPLIT = ""
# PATH_TRAIN_TEST_SPLIT_SAVE = "train_test_split_segmented.csv"

print(PATH_RADIOMICS)
print(PATH_TRAIN_TEST_SPLIT)


df = pd.read_csv(PATH_RADIOMICS, index_col=0)
print(df["shape"].unique())
# print(df["PixelSpacing"].unique())
if "PixelSpacing" in df.columns:
    cdrop = ["shape", "PixelSpacing"]
else:
    cdrop = ["shape"]

X = df.drop(columns=cdrop).dropna()

print(X.shape)


y = load_outcome(df.index.values)
print(X.shape, y.shape)
# print(y)


if PATH_TRAIN_TEST_SPLIT:
    split = pd.read_csv(PATH_TRAIN_TEST_SPLIT, index_col=0)
    idx_train = split[split == "train"].dropna().index
    idx_test = split[split == "test"].dropna().index
    idx_train = [idx.split(".")[0] for idx in idx_train]
    idx_test = [idx.split(".")[0] for idx in idx_test]
    # print(len(idx_train), len(idx_test))
    del split

    # For when extraction not finished for all patients
    print("LOADED train / test = ", len(idx_train), len(idx_test))
    idx_train = list(set(idx_train).intersection(set(y.index)))
    idx_test = list(set(idx_test).intersection(set(y.index)))
    print("\tREDUCED", len(idx_train), len(idx_test))

    y_train = y.loc[idx_train]
    y_test = y.loc[idx_test]
    X_train = X.loc[idx_train]
    X_test = X.loc[idx_test]

else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # save train / test split
    split = pd.Series(index=X.index.values)
    split.loc[X_train.index.values] = "train"
    split.loc[X_test.index.values] = "test"

    # print(split)
    split.to_csv(PATH_TRAIN_TEST_SPLIT_SAVE)
    print("Train / test split", split.shape, "saved at", PATH_TRAIN_TEST_SPLIT_SAVE)
    sys.exit()


print(y_train.shape, y_test.shape)
print(X_train.shape, X_test.shape)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train.shape, y_test.shape)

num_per_class = np.unique(y_train, return_counts=True)
print(f"per class TRAIN: N_{num_per_class[0][0]}={num_per_class[1][0]} ({num_per_class[1][0] / len(y_train) * 100 :.1f}%) / "
      f"N_{num_per_class[0][1]}={num_per_class[1][1]} ({num_per_class[1][1] / len(y_train) * 100 :.1f}%)")
num_per_class = np.unique(y_test, return_counts=True)
print(f"per class TEST: N_{num_per_class[0][0]}={num_per_class[1][0]} ({num_per_class[1][0] / len(y_test) * 100 :.1f}%) / "
      f"N_{num_per_class[0][1]}={num_per_class[1][1]} ({num_per_class[1][1] / len(y_test) * 100 :.1f}%)")


rf = RandomForestClassifier(min_samples_split=10, max_depth=20)
lr = LogisticRegression(max_iter=int(1e3), penalty="elasticnet", solver="saga", l1_ratio=0.9, C=1.0)
xgb = XGBClassifier()


models = {"LR":lr, "RF":rf, "XGB":xgb}
# models = {"RF1":rf}
# models = {"XGB":xgb}
# models = {"LR":lr}

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(data=scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(data=scaler.transform(X_test), index=X_test.index, columns=X_test.columns)


# To speed up pipeline for depelopment
# X_train_scaled = X_train_scaled.iloc[:100, :]
# X_test_scaled = X_test_scaled.iloc[:100, :]
# y_train = y_train.head(100)
# y_test = y_test.head(100)
print(X_train_scaled.shape, X_test_scaled.shape)
print(y_train.shape, y_test.shape)


# print(X_train)
# print(X_train_scaled)

for md, m in models.items():

    # rfe_scores = pd.DataFrame()

    # FEATURE SELECTION
    rfe = RFECV(estimator=m, step=10, cv=KFold(3), scoring="f1", min_features_to_select=1, n_jobs=-1, verbose=1)
    rfe.fit(X_train_scaled, y_train)
    dump(rfe, f"rfecv_{md}.joblib")
    print("SAVED RFE in", f"rfecv_{md}.joblib")

    # rfe = load(f"rfecv_{md}.joblib")


    fts = rfe.feature_names_in_[rfe.support_]
    print(len(rfe.cv_results_["mean_test_score"]), len(rfe.ranking_), np.shape(fts))
    # fts = X.columns.values  # no FS

    X_train = X_train_scaled.loc[:, fts]
    X_test = X_test_scaled.loc[:, fts]


    m.fit(X_train, y_train)
    yhat_train = m.predict(X_train)
    yhat = m.predict(X_test)
    # yhat = [0.5] * len(X_test)

    f1 = f1_score(y_test, yhat)
    acc = accuracy_score(y_test, yhat)
    auc = roc_auc_score(y_test, yhat)
    prc = precision_score(y_test, yhat)
    rec = recall_score(y_test, yhat)

    print(md, "Acc / AUC / Precision / Recall / F1 = ", np.round([acc, auc, prc, rec, f1], 3), end="\t")

    # f1 = f1_score(y_train, yhat_train)
    # acc = accuracy_score(y_train, yhat_train)
    # auc = roc_auc_score(y_train, yhat_train)
    # rec = recall_score(y_train, yhat_train)
    # print(np.round([acc, auc, rec, f1], 3))


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
    else:
        # print()
        pass