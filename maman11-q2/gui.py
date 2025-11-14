from __future__ import annotations
from numba import njit
import numpy as np
import pyqtgraph as pg
try:
    from PyQt6 import QtWidgets, QtCore, QtGui
    from pyqtgraph.Qt import mkQApp
except ImportError:
    from pyqtgraph.Qt import QtWidgets, QtCore, QtGui, mkQApp
# from PyQt6 import QtWidgets, QtCore, QtGui
# from pyqtgraph.Qt import mkQApp
import pyqtgraph.opengl as gl
import EcoWorld as ew

class MainWindow (QtWidgets.QMainWindow):
    """pyqtgraph-based GUI"""
    def __init__(self, world: ew.EcoWorld):
        # initialize simulation variables
        self.world = world
        # initialize GUI
        super(MainWindow, self).__init__()
        win = pg.GraphicsLayoutWidget(show=True)
        self.setCentralWidget(win)
        self.setWindowTitle("Pollution CA simulation")
        self.show()
        # set map
        self.map_view = win.addViewBox(lockAspect=True)
        self.map = pg.ImageItem()
        tr = QtGui.QTransform().translate(-0.5, -0.5)
        self.map.setTransform(tr)
        # vvvvv
        self.map.setImage(self.world.get_map())
        self.map_plot_item = win.addPlot()
        self.map_plot_item.addItem(self.map)
        self.map_plot_item.showAxes(False)
        # trackers
        # sea level


def run(world: ew.EcoWorld) -> None:
    mkQApp("mkQApp title")
    main_window = MainWindow(world)
    pg.exec()