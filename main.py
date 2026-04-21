import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
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
    bcwds["mean_area"].plot.hist(bins=20, grid=True)
    bcwds.plot.scatter("mean_area", "mean_concave_pts")
    
    # meanAreaSpread.plot.hist(grid=True, bins=20)
    
    # bMSpread.plot.pie()
    
    # iloc [row number, coulmn number]
    print(bcwds.iloc[:, 1:11].describe())
    
    print(bMSpread)
    
    plt.show()

if __name__ == "__main__":
    main()