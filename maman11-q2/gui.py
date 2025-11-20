from __future__ import annotations
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore, QtGui
from pyqtgraph.Qt import mkQApp, QtCore
from pyqtgraph.widgets import RemoteGraphicsView, GraphicsLayoutWidget
# import pyqtgraph.opengl as gl
import EcoWorld as ew

class MainWindow (QtWidgets.QMainWindow):
    """pyqtgraph-based GUI"""
    def update(self) -> None:
        if not self.paused:
            next(self.world) # advance world
        self.map.setImage(self.world.get_map()) # set map
        # set sea level plot
        self.sea_level_plot.plot(self.world.sea_level_history(), clear=True, _callSync='off')
        self.pollution_plot.plot(self.world.pollution_history(), clear=True, _callSync='off')
        self.temp_plot.plot(self.world.temperature_history(), clear=True, _callSync='off')

    def pause_func(self):
        self.paused = not self.paused
        self.pause_toggle.setText("Resume" if self.paused else "Pause")

    def set_iterations_per_second(self, ips: float):
        self.milliseconds_per_iteration = 1000.0 / ips
        # TODO: finish implementing iterations per second some other time

    def __init__(self, world: ew.EcoWorld):
        # initialize simulation variables
        self.world = world
        self.milliseconds_per_iteration = 0 # '0' means we update as fast as possible
        self.paused = False
        # initialize GUI
        super(MainWindow, self).__init__()
        win = pg.GraphicsLayoutWidget(show=True)
        self.setCentralWidget(win)
        self.setWindowTitle("Pollution CA simulation")
        self.resize(1920, 1080)
        self.show()

        # set map
        self.map_layout = win.addLayout(row=0, col=1)
        self.map_layout.addLabel("world map")
        self.map_layout.nextRow()
        self.map_view = self.map_layout.addViewBox(lockAspect=True)
        self.map = pg.ImageItem()
        self.map.setImage(self.world.get_map())
        self.map_view.addItem(self.map)
        # trackers
        self.tracker_layout = win.addLayout(row=0, col=0)
        self.tracker_layout.addLabel("Data")
        self.tracker_layout.setMaximumWidth(600)
        # sea level
        self.tracker_layout.nextRow()
        self.sea_level_plot = self.tracker_layout.addPlot(title="Average sea level")
        self.sea_level_plot.setClipToView(True)
        self.sea_level_plot.setDownsampling(mode='peak')
        self.sea_level_curve = self.sea_level_plot.plot(self.world.sea_level_history())
        # total pollution
        self.tracker_layout.nextRow()
        self.pollution_plot = self.tracker_layout.addPlot(title="Total pollution")
        self.pollution_plot.setClipToView(True)
        self.pollution_plot.setDownsampling(mode='peak')
        self.pollution_curve = self.pollution_plot.plot(self.world.pollution_history())
        # average temperature
        self.tracker_layout.nextRow()
        self.temp_plot = self.tracker_layout.addPlot(title="Average Temperature")
        self.temp_plot.setClipToView(True)
        self.temp_plot.setDownsampling(mode='peak')
        self.temp_curve = self.temp_plot.plot(self.world.temperature_history())

        # options
        self.opt_dock = QtWidgets.QDockWidget("Options", self)
        self.opt_widget = QtWidgets.QWidget()
        self.opt_layout = QtWidgets.QVBoxLayout(self.opt_widget)
        self.opt_layout.setContentsMargins(8, 8, 8, 8)
        # pause button
        self.pause_toggle = QtWidgets.QPushButton("Pause")
        self.pause_toggle.clicked.connect(self.pause_func)
        self.opt_layout.addWidget(self.pause_toggle)
        # surface toggle
        self.surface_toggle = QtWidgets.QCheckBox("show Surface")
        self.surface_toggle.setChecked(self.world.show_surface)
        self.surface_toggle.toggled.connect(self.world.surface_toggle)
        self.opt_layout.addWidget(self.surface_toggle)
        # pollution toggle
        self.pollution_toggle = QtWidgets.QCheckBox("show pollution")
        self.pollution_toggle.setChecked(self.world.show_pollution_toggle)
        self.pollution_toggle.toggled.connect(self.world.pollution_toggle)
        self.opt_layout.addWidget(self.pollution_toggle)
        # temperature toggle
        self.temperature_toggle = QtWidgets.QCheckBox("show temperature")
        self.temperature_toggle.setChecked(self.world.show_temperature_toggle)
        self.temperature_toggle.toggled.connect(self.world.temperature_toggle)
        self.opt_layout.addWidget(self.temperature_toggle)
        # cloud toggle
        self.clouds_toggle = QtWidgets.QCheckBox("show clouds")
        self.clouds_toggle.setChecked(self.world.show_clouds_toggle)
        self.clouds_toggle.toggled.connect(self.world.clouds_toggle)
        self.opt_layout.addWidget(self.clouds_toggle)
        # set options dock in window
        self.opt_layout.addStretch(1)
        self.opt_dock.setWidget(self.opt_widget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.opt_dock)

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