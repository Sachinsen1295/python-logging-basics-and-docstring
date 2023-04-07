import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import joblib
from matplotlib.colors import ListedColormap
import logging

plt.style.use("fivethirtyeight")


def prepare_date(df, target_col ="y"):
    """it returns Label and Independent features

    Args:
        df (pd.Dataframe): This is the DataFrame
        target_col (str, optional): label column name is given. Defaults to "y".

    Returns:
        tuple: label and x
    """

    logging.info("Prepaing the data for training")
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    return X,y



def save_plot(df, model, filename="plot.png", plot_dir="plots"):
    def _create_base_plot(df):

        logging.info("Creating the base plot")

        df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="coolwarm")
        plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
        
        figure = plt.gcf()
        figure.set_size_inches(10, 8)
    
    def _plot_decision_regions(X, y, classifier, resolution=0.02):

        logging.info("Ploting the decision regions")

        colors = ("cyan", "lightgreen")
        cmap = ListedColormap(colors)
        
        X = X.values # as an array
        x1 = X[:, 0]
        x2 = X[:, 1]
        
        x1_min, x1_max = x1.min() - 1, x1.max() + 1 
        x2_min, x2_max = x2.min() - 1, x2.max() + 1
        
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution)
                              )
        y_hat = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        y_hat = y_hat.reshape(xx1.shape)
        
        plt.contourf(xx1, xx2, y_hat, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        
        plt.plot()


    X,y = prepare_date(df)

    _create_base_plot(df)
    _plot_decision_regions(X,y,model)

    os.makedirs(plot_dir, exist_ok=True) #to create directory
    plot_path = os.path.join(plot_dir,filename)
    plt.savefig(plot_path)
    logging.info(f"saving the plot at {plot_path}")


