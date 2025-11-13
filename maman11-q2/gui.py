from __future__ import annotations
from numba import njit
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui, mkQApp
import pyqtgraph.opengl as gl
import EcoWorld as ew

class MainWindow (QtWidgets.QMainWindow):
    """pyqtgraph-based GUI"""
    def __init__(self, world: ew.EcoWorld):
        # initialize simulation variables
        self.world = world
        # initialize GUI
        super(MainWindow, self).__init__()
        gr_wid = pg.GraphicsLayoutWidget(show=True)
        self.setCentralWidget(gr_wid)
        self.setWindowTitle("Pollution CA simulation")
        self.show()
        # set grid




def run(world:ew.EcoWorld) -> None:
    mkQApp("mkQApp title")
    main_window = MainWindow(world)
    pg.exec()