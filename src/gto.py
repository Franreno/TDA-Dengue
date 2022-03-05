"""
Creating same plots using giotto's mapper.
"""

# Data handling
import numpy as np
import pandas as pd

# Data vizualization
from gtda.plotting import plot_point_cloud

from gtda.mapper import (
    CubicalCover,
    make_mapper_pipeline,
    Projection,
    plot_static_mapper_graph,
    plot_interactive_mapper_graph,
    MapperInteractivePlotter
)

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from main import GenerateVectors

from sklearn import datasets


DengueDataCSV = './data/strict-data/2010DengueData.csv'

DengueDataFrame = pd.read_csv(DengueDataCSV)

data, _ = datasets.make_circles(n_samples=5000, noise=0.05, factor=0.3, random_state=42)

plot_point_cloud(data)
