
import datetime as dt, pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor
from google.colab import drive

drive.mount("/content/drive")
forest = pd.read_csv("drive/kongkea/Dataset/fire.csv")
forest.head()
forest.shape
forest.isnull().sum()
forest.describe()
plt.figure(figsize=(10, 10))
sns.heatmap(forest.corr(), annot=True, cmap="viridis", linewidths=0.5)
forest = forest.drop(["track"], axis=1)
print("scan")
print(forest["scan"].value_counts())
print()
print("aqc_time")
print(forest["acq_time"].value_counts())
print()
print("satellite")
print(forest["satellite"].value_counts())
print()
print("instrument")
print(forest["instrument"].value_counts())
print()
print("version")
print(forest["version"].value_counts())
print()
print("daynight")
print(forest["daynight"].value_counts())
print()
forest = forest.drop(["instrument", "version"], axis=1)
forest.head()
daynight_map = {"D": 1, "N": 0}
satellite_map = {"Terra": 1, "Aqua": 0}
forest["daynight"] = forest["daynight"].map(daynight_map)
forest["satellite"] = forest["satellite"].map(satellite_map)
forest.head()
forest["type"].value_counts()
types = pd.get_dummies(forest["type"])
forest = pd.concat([forest, types], axis=1)
forest = forest.drop(["type"], axis=1)
forest.head()
forest = forest.rename(columns={0: "type_0", 2: "type_2", 3: "type_3"})
bins = [0, 1, 2, 3, 4, 5]
labels = [1, 2, 3, 4, 5]
forest["scan_binned"] = pd.cut(forest["scan"], bins=bins, labels=labels)
forest.head()
forest["acq_date"] = pd.to_datetime(forest["acq_date"])
forest = forest.drop(["scan"], axis=1)
forest["year"] = forest["acq_date"].dt.year
forest.head()
forest["month"] = forest["acq_date"].dt.month
forest["day"] = forest["acq_date"].dt.day
forest.shape
y = forest["confidence"]
fin = forest.drop(
    ["confidence", "acq_date", "acq_time", "bright_t31", "type_0"], axis=1
)
plt.figure(figsize=(10, 10))
sns.heatmap(fin.corr(), annot=True, cmap="viridis", linewidths=0.5)
fin.head()
Xtrain, Xtest, ytrain, ytest = train_test_split(fin.iloc[:, :500], y, test_size=0.2)
random_model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
random_model.fit(Xtrain, ytrain)
y_pred = random_model.predict(Xtest)
random_model_accuracy = round(random_model.score(Xtrain, ytrain) * 100, 2)
print(round(random_model_accuracy, 2), "%")
random_model_accuracy1 = round(random_model.score(Xtest, ytest) * 100, 2)
print(round(random_model_accuracy1, 2), "%")
import pickle

saved_model = pickle.dump(
    random_model, open("drive/kongkea/Dataset/Models/ForestModelOld.pickle", "wb")
)
random_model.get_params()
from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start=300, stop=500, num=20)]
max_features = ["auto", "sqrt"]
max_depth = [int(x) for x in np.linspace(15, 35, num=7)]
max_depth.append(None)
min_samples_split = [2, 3, 5]
min_samples_leaf = [1, 2, 4]
random_grid = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
}
print(random_grid)
rf_random = RandomizedSearchCV(
    estimator=random_model,
    param_distributions=random_grid,
    n_iter=50,
    cv=3,
    verbose=2,
    random_state=42,
)
rf_random.fit(Xtrain, ytrain)
rf_random.best_params_
random_new = RandomForestRegressor(
    n_estimators=394,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    max_depth=25,
    bootstrap=True,
)
random_new.fit(Xtrain, ytrain)
y_pred1 = random_new.predict(Xtest)
random_model_accuracy1 = round(random_new.score(Xtrain, ytrain) * 100, 2)
print(round(random_model_accuracy1, 2), "%")
random_model_accuracy2 = round(random_new.score(Xtest, ytest) * 100, 2)
print(round(random_model_accuracy2, 2), "%")
saved_model = pickle.dump(
    random_new, open("drive/kongkea/Dataset/Models/forest_fire_model.pickle", "wb")
)
import bz2

compressionLevel = 9
source_file = "drive/kongkea/Dataset/Models/forest_fire_model.pickle"
destination_file = "drive/kongkea/Dataset/Models/forest_fire_model.bz2"
with open(source_file, "rb") as data:
    tarbz2contents = bz2.compress(data.read(), compressionLevel)
fh = open(destination_file, "wb")
fh.write(tarbz2contents)
fh.close()
