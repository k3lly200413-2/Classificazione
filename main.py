import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from os import path
from urllib.request import urlretrieve


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
    
    # benign and malign or whatever
    diagnosis_colour_map = {"B": "blue", "M": "red"}
    
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
    
    print(sep_x2)
    
    X2d_train.plot.scatter(
        "mean_area", 
        "mean_concave_pts", 
        c=y_train.map(diagnosis_colour_map)
    )
    
    plt.plot(sep_x1, sep_x2, c="green", linewidth=2)
        
    y_pred = np.where(X2d_train["mean_concave_pts"] > -0.0001 * X2d_train["mean_area"] + 0.15, "M", "B")

    print(y_pred)

    plt.show()

if __name__ == "__main__":
    main()