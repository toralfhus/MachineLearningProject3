from utils import *
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score


PATH_RADIOMICS = "radiomics_fbc=32_train.csv"
df = pd.read_csv(PATH_RADIOMICS, index_col=0)
print(df["shape"].unique())
print(df["PixelSpacing"].unique())
X = df.drop(columns=["shape", "PixelSpacing"])
print(X.shape)


y = load_outcome(df.index.values)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_test.shape)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
yhat = rf.predict(X_test)

f1 = f1_score(y_test, yhat)
acc = accuracy_score(y_test, yhat)
auc = roc_auc_score(y_test, yhat)
rec = recall_score(y_test, yhat)

print("Acc / AUC / Recall / F1 = ", np.round([acc, auc, rec, f1], 3))

ft_importances = {ft:imp for ft, imp in zip(rf.feature_names_in_, rf.feature_importances_)}
ft_importances = dict(sorted(ft_importances.items(), key=lambda x:x[1], reverse=True))


plt.plot(range(len(ft_importances)), ft_importances.values(), "x")
plt.show()
