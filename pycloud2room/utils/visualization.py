import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry.base import BaseGeometry
from geopandas import GeoSeries

from typing import Union, List


def plot_geometry(geometry: Union[GeoSeries, List[BaseGeometry], BaseGeometry],
                  ax: plt.Artist = None):
    if isinstance(geometry, list):
        series = GeoSeries(geometry)
    elif isinstance(geometry, GeoSeries):
        series = GeoSeries
    else:
        series = GeoSeries([geometry])
    if ax is None:
        fig, ax = plt.subplots()
    series.plot(ax=ax)


def plt_imshow(img):
    print(img.shape)
    plt.imshow(img)
    plt.show()


def plot_img_rgb(img):
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    print(img.shape)
    plt.imshow(img)
    plt.show()


def plot_img_gray(img):
    print(img.shape)
    plt.imshow(img)
    plt.show()
