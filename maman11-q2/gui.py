from __future__ import annotations
from numba import njit
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore, QtGui
from pyqtgraph.Qt import mkQApp
from pyqtgraph.widgets import RemoteGraphicsView, GraphicsLayoutWidget
import pyqtgraph.opengl as gl
import EcoWorld as ew

class MainWindow (QtWidgets.QMainWindow):
    """pyqtgraph-based GUI"""
    def update(self) -> None:
        next(self.world) # advance world
        self.map.setImage(self.world.get_map()) # set map
        # set sea level plot
        # self.sea_level_curve.setData(self.world.sea_level_history())
        self.sea_level_plot.plot(self.world.sea_level_history(), clear=True, _callSync='off')

    def __init__(self, world: ew.EcoWorld):
        # initialize simulation variables
        self.world = world
        self.milliseconds_per_iteration = 0 # '0' means we update as fast as possible
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
        self.sea_level_plot = self.tracker_layout.addPlot(title="Average sea level")
        # self.sea_level_plot.setClipToView(True)
        # self.sea_level_plot.setDownsampling(mode='peak')
        self.sea_level_curve = self.sea_level_plot.plot(self.world.sea_level[:self.world.sea_level_ptr])

        # cross-hair, for inspecting each cell's value
        # TODO
        # timer (for plot update)
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.milliseconds_per_iteration)



def run(world: ew.EcoWorld) -> None:
    mkQApp("mkQApp title")
    main_window = MainWindow(world)
    pg.exec()