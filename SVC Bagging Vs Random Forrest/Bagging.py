import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#load dataset
df = pd.read_csv('csgo_round_snapshots.csv')
#get map names as labels
mle = LabelEncoder()
df["map"] = mle.fit_transform(df["map"])
#get numerical columns
X = df.iloc[:, [0, 1, 2] + list(range(5, 18))]
y = df["map"]

# normalizing X for SVC to work
X -= np.average(X, axis=0)
X /= np.std(X, axis=0)

# using a linear SVC and bagging with 100 estimators
clf = LinearSVC()
bagging_svc = BaggingClassifier(estimator=clf, n_estimators=100, n_jobs=-1, oob_score=True, verbose=3)
bagging_svc.fit(X, y)

rf_params = {"max_depth": [1, 2, 3, 4 ,5, 10, 15, 20, 30]}
rfc_clf = RandomForestClassifier(n_estimators=100, oob_score=True)
grid_search_rf = GridSearchCV(rfc_clf, param_grid=rf_params, cv=5, n_jobs=-1, verbose=3)

grid_search_rf.fit(X, y)


print("Best depth:", grid_search_rf.best_params_["max_depth"])
print("-----------------")
print("bagging SVC:", bagging_svc.score(X, y))
print("Bagging OOB:", bagging_svc.oob_score_)
print("Random Forest:", grid_search_rf.best_estimator_.score(X, y))
print("RF OOB:", grid_search_rf.best_estimator_.oob_score_)

bagging_predict = bagging_svc.predict(X)
rf_predict = grid_search_rf.best_estimator_.predict(X)

cm = confusion_matrix(y, bagging_predict, normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=mle.classes_)
disp_cm.plot()

cm_rf = confusion_matrix(y, rf_predict, normalize='true')
disp_cm_rf = ConfusionMatrixDisplay(cm_rf, display_labels=mle.classes_)
disp_cm_rf.plot()

plt.show()