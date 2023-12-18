from utils import *
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score
import seaborn as sns
from matplotlib import pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras
#from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier
from keras import regularizers

PATH_RADIOMICS = "radiomics_seg_fbc=32_train.csv"
#PATH_RADIOMICS = "radiomics_fbc=32_train.csv"
df = pd.read_csv(PATH_RADIOMICS, index_col=0)
print(df["shape"].unique())
#print(df["PixelSpacing"].unique())
#X = df.drop(columns=["shape", "PixelSpacing"]).dropna()
X = df.drop(columns=["shape"]).dropna()

print(X.shape)

y = load_outcome(df.index.values)
print(X.shape, y.shape)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_test.shape)

# Define the neural network model with L2 regularization
lmbd_vals = np.logspace(-3, 0, 4)
lr_vals = np.logspace(-3, 0, 4)
optimizer_list = np.array(['Adam'])
accuracy_list = np.zeros((len(lmbd_vals), len(lr_vals)))

# Search for optimal parameters
"""
for k, optimizer in enumerate(optimizer_list):

    for i, lmbd in enumerate(lmbd_vals):
        for j, lr in enumerate(lr_vals):

            adam = keras.optimizers.legacy.Adam(learning_rate=lr)
            rmsprop  = keras.optimizers.legacy.RMSprop(learning_rate=lr)
            sgd  = keras.optimizers.legacy.SGD(learning_rate=lr)
            adagrad = keras.optimizers.legacy.Adagrad(learning_rate=lr)

            optimizer_names = {
                'Adam': adam,
                'Adagrad': adagrad,
                'SGD': sgd,
                'RMSprop': rmsprop} 

            selected_optimizer = optimizer_names.get(optimizer, None)

            if selected_optimizer is None:
                print(f"404 Optimizer '{optimizer}' not found.")
            else:
                optimizer = selected_optimizer

            def create_model():
                model = Sequential()
                model.add(Dense(8, input_dim=1023, activation='relu', kernel_regularizer=regularizers.l2(lmbd)))
                model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(lmbd)))
                model.add(Dense(1, activation='sigmoid'))
                model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                return model

            print(f'Now at Lambda = {lmbd}, lr = {lr}, Optimizer = {optimizer}')

            # Evaluate model using cross-validation 
            model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=10, verbose=0)
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_results = cross_val_score(model, X_train, y_train, cv=kfold)
            print("Cross-Validation Accuracy: %.2f%% (+/- %.2f%%)" % (cv_results.mean() * 100, cv_results.std() * 100))

            # Train the final model on the full training set
            final_model = create_model()
            final_model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=0)

            # Evaluate the final model on the test set
            y_pred = final_model.predict(X_test)
            y_pred_classes = np.round(y_pred).flatten().astype(int)
            test_accuracy = accuracy_score(y_test, y_pred_classes)
            accuracy_list[i][j] = test_accuracy
            print("Test Accuracy: %.2f%%" % (test_accuracy * 100))

    # Create accuracy heatmap for finding lambda and learning rate
    sns.heatmap(accuracy_list, annot=True, cmap="YlGnBu", cbar=True)
    plt.ylabel("Lambda values in log-scale")
    plt.xlabel("Learning rate")
    plt.title(f"Accuracy Matrix for {optimizer_list[k]}")
    plt.savefig(f'Accuracy {optimizer_list[k]}.pdf')
    #plt.show()
"""

# Final Model Training and Evaluation

#rmsprop  = keras.optimizers.legacy.RMSprop(learning_rate=0.1)
adam = keras.optimizers.legacy.Adam(learning_rate=0.1)

def create_model():
    model = Sequential()
    model.add(Dense(8, input_dim=1023, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model

# Evaluate model using cross-validation 
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_val_score(model, X_train, y_train, cv=kfold)
print("Cross-Validation Accuracy: %.2f%% (+/- %.2f%%)" % (cv_results.mean() * 100, cv_results.std() * 100))

# Train the final model on the full training set
final_model = create_model()
final_model.fit(X_train, y_train, epochs=1000, batch_size=10, verbose=0)

# Evaluate the final model on the test set
y_pred = final_model.predict(X_test)
y_pred_classes = np.round(y_pred).flatten().astype(int)
print(len(y_pred_classes))

test_accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes)
recall = recall_score(y_test, y_pred_classes)
brier = brier_score_loss(y_test, y_pred)
f1 = f1_score(y_test, y_pred_classes, average='binary')

print('Test Accuracy: %.2f%%' % (test_accuracy * 100))
print(f'Brier Score {brier}')
print(f'Recall Score {recall}')
print(f'Precision Score {precision}')
print(f'F1 Score {f1}')

cm = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:")
print(cm)

# Calculate AUC-ROC score
auc_roc = roc_auc_score(y_test, y_pred)
print(f'AUC Score: {auc_roc}')

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC-curve' ,color='firebrick')
plt.plot([0, 1], [0, 1], linestyle='-', color='forestgreen', label='Random')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.legend()
plt.savefig('ROC.pdf')
plt.show()
