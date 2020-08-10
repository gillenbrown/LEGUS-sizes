"""
open_some_fits.py - this is basically the select PSF stars script, only without the
buttons to approve or reject. It shows 10 randomly selected clusters per field
"""
import shutil

import numpy as np
from astropy import table
from pathlib import Path
import random

from PySide2.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QShortcut,
)
from PySide2.QtGui import QPixmap, QKeySequence
from PySide2.QtCore import Qt

# ======================================================================================
#
# Find the location of the data, and get 10 random plots for each field
#
# ======================================================================================
def criteria(catalog):
    mask_1 = catalog["axis_ratio_best"] > 0
    mask_2 = catalog["axis_ratio_best"] == 1.0
    mask = np.logical_and(mask_1, mask_2)
    catalog["check_this"] = mask


plots_to_show = []
data_dir = Path("../data").resolve()
for item in sorted(data_dir.iterdir()):
    if item.is_dir():
        size_dir = item / "size"
        fits_dir = size_dir / "cluster_fit_plots"
        # open the catalog, and get the ones that match our criteria
        cat_name = "final_catalog_30_pixels_psf_my_stars_15_pixels_2x_oversampled.txt"
        cat = table.Table.read(size_dir / cat_name, format="ascii.ecsv")
        criteria(cat)

        ids_to_check = cat["ID"][cat["check_this"]].data

        these_plots = []
        for plot in fits_dir.iterdir():
            name = plot.name
            if name.endswith("size_30.png"):
                this_id = int(name.split("_")[1])
                if this_id in ids_to_check:
                    these_plots.append(plot)
        # then select a random 10
        if len(these_plots) > 10:
            these_plots = random.sample(these_plots, 10)

        plots_to_show += these_plots

plots_to_show = [str(item) for item in plots_to_show]
# ======================================================================================
#
# Setting up GUI
#
# ======================================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.idx = 0

        # the layout of this will be as follows: There will be an image of the cluster
        # shown Below that will be two buttons: previous and next
        vBoxMain = QVBoxLayout()

        self.label = QLabel()
        vBoxMain.addWidget(self.label)

        self.image = QLabel()
        self.image.setScaledContents(True)
        vBoxMain.addWidget(self.image)

        # then the buttons at the bottom, which will be laid our horizontally
        hBoxInput = QHBoxLayout()
        self.previousButton = QPushButton("Previous")
        self.nextButton = QPushButton("Next")
        self.copyButton = QPushButton("Copy To Desktop, then Next")
        # set the tasks that each button will do
        self.previousButton.clicked.connect(self.previous)
        self.nextButton.clicked.connect(self.next)
        self.copyButton.clicked.connect(self.copy)
        # and make keyboard shortcuts
        previousShortcut = QShortcut(QKeySequence("left"), self.previousButton)
        nextShortcut = QShortcut(QKeySequence("right"), self.nextButton)
        copyShortcut = QShortcut(QKeySequence("c"), self.copyButton)

        previousShortcut.activated.connect(self.previous)
        nextShortcut.activated.connect(self.next)
        copyShortcut.activated.connect(self.copy)

        hBoxInput.addWidget(self.previousButton)
        hBoxInput.addWidget(self.nextButton)
        hBoxInput.addWidget(self.copyButton)
        vBoxMain.addLayout(hBoxInput)

        # have to set a dummy widget to act as the central widget
        container = QWidget()
        container.setLayout(vBoxMain)
        self.setCentralWidget(container)

        # add the first star
        self.showCluster()

        self.resize(800, 800)

        # then we can show the widget
        self.show()

    def next(self):
        self.idx += 1
        self.showCluster()

    def previous(self):
        self.idx -= 1
        self.showCluster()

    def copy(self):
        shutil.copy2(plots_to_show[self.idx], Path.home() / "Desktop")
        self.next()

    def showCluster(self):
        # add the appropriate image to the GUI
        if self.idx >= len(plots_to_show):
            QApplication.quit()
        pixmap = QPixmap(plots_to_show[self.idx])
        pixmap = pixmap.scaled(800, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image.setPixmap(pixmap)
        self.image.setScaledContents(True)
        self.image.repaint()

        # and adjust the text
        self.label.setText(Path(plots_to_show[self.idx]).name)
        self.label.repaint()


app = QApplication()

# The MainWindow class holds all the structure
window = MainWindow()

# Execute application
app.exec_()
