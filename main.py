import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from os import path

from urllib.request import urlretrieve

# benign and malign or whatever
diagnosis_colour_map = {"B": "blue", "M": "red"}

def separator_2d(model, x1):
    # ricaviamo w e b dal modello
    w = model.coef_[0]
    b = model.intercept_[0]
    # riportiamo in NumPy la formula sopra
    return -x1 * w[0] / w[1] - b / w[1]

def plot_separator_on_data(X, y, model=None):
    X = np.array(X)
    colors = pd.Series(y).map(diagnosis_colour_map)
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    if model is not None:
        xlim, ylim = plt.xlim(), plt.ylim()
        # xlim is a tuple, *xlim unpacks them into two values
        sep_x = np.linspace(*xlim, 2)
        sep_y = separator_2d(model, sep_x)
        plt.plot(sep_x, sep_y, c="green", linewidth=2)
        plt.xlim(xlim); plt.ylim(ylim)


def main():
    # breast cancer wisconson data set
    BCWDS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"


    if not path.exists("wdbc.data"):
        urlretrieve(BCWDS_URL, "wdbc.data")
    bcwds = pd.read_csv("wdbc.data", header=None, index_col=0)
    
    # print(bcwds.head(3))
    
    cell_features = [
        "radius",     "texture",
        "perimeter",  "area",
        "smoothness", "compactness",
        "concavity",  "concave_pts",
        "symmetry",   "fractal_dim",
    ]
    
    stats = ["mean", "std", "worst"]
    
    bcwds.columns = ["diagnosis"] + [f"{stat}_{feat}" for stat in stats for feat in cell_features]

    pd.options.display.max_columns = 31
    
    # print(bcwds.head(5))
    
    bMSpread = bcwds["diagnosis"].value_counts()
    # bcwds["mean_area"].plot.hist(bins=20, grid=True)
    # bcwds.plot.scatter("mean_area", "mean_concave_pts")
    
    # meanAreaSpread.plot.hist(grid=True, bins=20)
    
    # bMSpread.plot.pie()
    
    # iloc [row number, coulmn number]
    # print(bcwds.iloc[:, 1:11].describe())
    
    # print(bMSpread)
    
    # bcwds.pivot_table(
    #     index=bcwds.index, 
    #     columns="diagnosis", 
    #     values="mean_area"
    #     ).plot.hist(
    #         bins=20, 
    #         stacked=True,       # Stacked makes columns stack one on top of another
    #         figsize=(12, 6)
    #     )
    
    diagnosis_colours = bcwds["diagnosis"].map(diagnosis_colour_map)
    
    # print(diagnosis_colours.tail(5))
    
    # bcwds.plot.scatter("mean_area", "mean_concave_pts", c=diagnosis_colours)
    
    y = bcwds["diagnosis"]
    X2d = bcwds[["mean_area", "mean_concave_pts"]]
    
    X2d_train, X2d_val, y_train, y_val = train_test_split(
        X2d, y,
        test_size=1/3,
        random_state=42
    )
    
    # X2d_train.plot.scatter(
    #     "mean_area", 
    #     "mean_concave_pts", 
    #     c=y_train.map(diagnosis_colour_map)
    # )
    
    sep_x1 = np.linspace(0, 1500, 2)
    
    sep_x2 = -0.0001 * sep_x1 + 0.15
    
    # print(sep_x2)
    
    # X2d_train.plot.scatter(
    #     "mean_area", 
    #     "mean_concave_pts", 
    #     c=y_train.map(diagnosis_colour_map)
    # )
    
    # plt.plot(sep_x1, sep_x2, c="green", linewidth=2)
    
    
    # X2d_val is used because we need to make predictions with data we have never seen before
    y_pred = np.where(X2d_val["mean_concave_pts"] > -0.0001 * X2d_val["mean_area"] + 0.15, "M", "B")

    # correct_class = np.array(y_pred == y_val)

    # print(correct_class.mean())

    scaler = StandardScaler()
    X2dn_train = scaler.fit_transform(X2d_train)
    X2dn_val = scaler.transform(X2d_val)
    
    # plt.scatter(X2dn_train[:, 0], X2dn_train[:, 1], c=y_train.map(diagnosis_colour_map))

    model = Perceptron(random_state=42)
    
    model.fit(X2dn_train, y_train)
    
    # # [0] is in reference to which hyperplane you wish to use 
    # print(model.coef_[0])
    
    # print(model.intercept_[0])

    # plot_separator_on_data(X2dn_val, y_val, model)
    
    # print(model.predict(X2dn_val[:3]))
    
    # print(model.score(X2dn_val, y_val))

    y_pred = model.predict(X2dn_val)
    cm = confusion_matrix(y_val, y_pred)
    
    # print(pd.DataFrame(cm, index=model.classes_, columns=model.classes_))
    # print(cm.diagonal().sum() / cm.sum())
    
    #       quante istanze del validation set sono PREDETTE "M"
    #                          ^^^^^^^^^^^^^^
    #          B    M
    #       B  122  0
    #       M  20   48
    malignant_prc = cm[1, 1] / cm[:, 1].sum()
    # print(malignant_prc)
    
    #        quante istanze del validation set sono REALMENTE "M"
    #                          ^^^^^^^^^^^^^^
    malignant_rec = cm[1, 1] / cm[1, :].sum()
    # print(malignant_rec)
    
    # f1-measure
    # print(2 * malignant_prc * malignant_rec / (malignant_prc + malignant_rec))

    model = LogisticRegression(
        solver="saga", 
        random_state=42
    )

    model.fit(X2dn_train, y_train)
    
    # print(model.coef_[0])
    
    # print(model.intercept_[0])
    
    # plot_separator_on_data(X2dn_train, y_train, model)
    
    # print(model.classes_)
    
    # print(model.predict_proba(X2dn_val[:3]))
    
    y_pred = model.predict(X2dn_val)
    
    # estraggo un campione di 100x100 punti nel piano
    mx1, mx2 = np.meshgrid(
        np.linspace(-2, 5.5, 100), 
        np.linspace(-2, 4.5, 100)
    )
    # estraggo le probabilità della classe M per ciascun punto
    # ravel just turns a multidimentional array into a 1-D array by just concatinating the inner arrays
    # my = model.predict_proba(
    #     np.c_[
    #         mx1.ravel(), 
    #         mx2.ravel()]
    #     )[:, 1].reshape(mx1.shape)
    # # disegno il grafico
    # plt.contourf(mx1, mx2, my, cmap="summer")
    # plt.scatter(*X2dn_train.T, c=y_train.map(diagnosis_colour_map))
    # plt.colorbar()
    
    # print(model.score(X2dn_val, y_val))
    
    # print(confusion_matrix(y_val, y_pred))
    
    # print(f1_score(y_val, model.predict(X2dn_val), pos_label="M"))

    # KFold — splits randomly without caring about class distribution. 
    # Each fold might end up with a very different ratio of classes by chance.
    # 
    # StratifiedKFold — splits so that each fold preserves the same class ratio 
    # as the original dataset.

    # THIS IS JUST FOR TESTING TO SEE THE DIFFERENCE BETWEEN THE TWO CLASSES

    kf = KFold(3, shuffle=True, random_state=42)
    
    # KFold
    # for train, val in kf.split(X2dn_train, y_train):
    #     print(y_train.iloc[val].value_counts())
        
    skf = StratifiedKFold(3, shuffle=True, random_state=42)

    # StratifiedKFold
    # for train, val in skf.split(X2dn_train, y_train):
    #     print(y_train.iloc[val].value_counts())
        
    model = LogisticRegression(solver="saga", random_state=42)
    
    # L2 and L1 cases would be repeated with l1_ratio that does not 
    # influence the outcome
    
    grid = {
        "penalty":  [ "l2", "l1", "elasticnet" ], # 3
        "C":        [ 0.1, 1, 10 ],               # 3
        "l1_ratio": [ 0.2, 0.5 ],                 # 2
    }
    
    grid = [
        {
            "penalty":  ["l2", "l1"],   # 2
            "C":        [0.1, 1, 10],   # 3
        },
        {
            "penalty":  ["elasticnet"], # 1
            "C":        [0.1, 1, 10],   # 3
            "l1_ratio": [0.2, 0.5],     # 2
        },
    ]
    
    gs = GridSearchCV(model, grid, cv=skf)
    
    gs.fit(X2dn_train, y_train)
    
    print(gs.best_params_)
    
    print(pd.DataFrame(gs.cv_results_).sort_values("rank_test_score").head(5))

    plt.show()

if __name__ == "__main__":
    main()