from __future__ import annotations
from numba import njit
import numpy as np
import pyqtgraph as pg
# try:
#     from PyQt6 import QtWidgets, QtCore, QtGui, QApplication
#     from pyqtgraph.Qt import mkQApp
# except ImportError:
#     from pyqtgraph.Qt import QtWidgets, QtCore, QtGui, mkQApp
from PyQt6 import QtWidgets, QtCore, QtGui
from pyqtgraph.Qt import mkQApp
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
        self.map_layout = win.addLayout()
        self.map_layout.addLabel("world map")
        self.map_layout.nextRow()
        self.map_view = self.map_layout.addViewBox(lockAspect=True)
        self.map = pg.ImageItem()
        self.map.setImage(self.world.get_map())
        self.map_view.addItem(self.map)
        # trackers
        self.tracker_layout = win.addLayout()
        self.tracker_layout.addLabel("Data")
        # sea level
        self.tracker_layout.nextRow()
        # self.sea_level_view = self.tracker_layout.addViewBox()
        self.sea_level = self.tracker_layout.addPlot(title="Average sea level")
        self.sea_level_curve = self.sea_level.plot(self.world.sea_level[:self.world.sea_level_ptr])

    def update(self) -> None:
        # TODO: implement update (apply next(world), update world & trackers


def run(world: ew.EcoWorld) -> None:
    mkQApp("mkQApp title")
    main_window = MainWindow(world)
    pg.exec()