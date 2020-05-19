"""
select_psf_stars.py

Select stars to be made into the PSF. This takes the outputs of
`preliminary_star_list.py` and presents these stars to the user for inspection.
"""
import sys
from pathlib import Path

from astropy import table
from astropy import stats
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import betterplotlib as bpl
from PySide2.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
)
from PySide2.QtGui import QPixmap
from PySide2.QtCore import Qt

import utils

bpl.set_style()

# ======================================================================================
#
# Get the parameters the user passed in
#
# ======================================================================================
# start by getting the output catalog name, which we can use to get the home directory
final_catalog = Path(sys.argv[1]).absolute()
home_dir = final_catalog.parent
# We'll need to get the preliminary catalog too
preliminary_catalog_path = Path(sys.argv[2]).absolute()

# ======================================================================================
#
# Get the data - preliminary catalog and images, plus other setup
#
# ======================================================================================
# read in this catalog
stars_table = table.Table.read(preliminary_catalog_path, format="ascii.ecsv")

# and the image
image_data, instrument = utils.get_f555w_drc_image(home_dir)

# get the noise_level, which will be used later
_, _, noise = stats.sigma_clipped_stats(image_data, sigma=2.0)

# Set the box size that will be used as the size of the PSF image. This is chosen as the
# size of the Ryon+ 17 fitting region. I'll check for duplicates within this region.
width = 31

# ======================================================================================
#
# Setting up GUI
#
# ======================================================================================
# make the temporary location to store images
temp_loc = str(Path(__file__).absolute().parent / "temp.png")
# and add a column to the table indicating which stars are kept
stars_table["use_for_psf"] = False


class MainWindow(QMainWindow):
    def __init__(self, starList, imageData, width):
        QMainWindow.__init__(self)

        self.starData = starList
        self.idx = -1
        self.imageData = imageData
        self.plotWidth = width

        # the layout of this will be as follows: There will be an image of the star
        # shown, with it's data on the left side. Below that will be two buttons:
        # yes and no
        vBoxMain = QVBoxLayout()

        # The first thing will be the data and image, which are laid out horizontally
        hBoxImageData = QHBoxLayout()
        self.image = QLabel()
        self.starDataText = QLabel("Image data here\nAttr 1\nAttr2")
        hBoxImageData.addWidget(self.image)
        hBoxImageData.addWidget(self.starDataText)
        hBoxImageData.setAlignment(Qt.AlignTop)
        vBoxMain.addLayout(hBoxImageData)

        # then the buttons at the bottom, which will also be laid our horizontally
        hBoxInput = QHBoxLayout()
        self.acceptButton = QPushButton("Accept")
        self.rejectButton = QPushButton("Reject")
        self.exitButton = QPushButton("Done Selecting Stars")
        # set the tasks that each button will do
        self.acceptButton.clicked.connect(self.accept)
        self.rejectButton.clicked.connect(self.reject)
        self.exitButton.clicked.connect(self.exit)

        hBoxInput.addWidget(self.acceptButton)
        hBoxInput.addWidget(self.rejectButton)
        hBoxInput.addWidget(self.exitButton)
        vBoxMain.addLayout(hBoxInput)

        # have to set a dummy widget to act as the central widget
        container = QWidget()
        container.setLayout(vBoxMain)
        self.setCentralWidget(container)
        # self.resize(1000, 1000)

        # add the first star
        self.nextStar()

        # then we can show the widget
        self.show()

    def accept(self):
        self.starData["use_for_psf"][self.idx] = True
        self.nextStar()

    def reject(self):
        self.starData["use_for_psf"][self.idx] = False
        self.nextStar()

    def exit(self):
        QApplication.quit()

    def nextStar(self):
        self.idx += 1
        thisStar = self.starData[self.idx]
        while thisStar["is_cluster"]:
            self.idx += 1
            thisStar = self.starData[self.idx]
        # make the temporary plot
        self.snapshot()
        # then add it to the the GUI
        self.image.setPixmap(QPixmap(temp_loc))
        # update the label
        new_info = (
            f"Number of Accepted Stars: {np.sum(self.starData['use_for_psf'])}\n\n"
            f"Index: {self.idx}\n"
            f"FWHM: {thisStar['fwhm']:.3f}\n"
            f"Roundness: {thisStar['roundness']:.3f}\n"
            f"Sharpness: {thisStar['sharpness']:.3f}\n\n"
        )
        if thisStar["near_star"]:
            new_info += "NEAR ANOTHER STAR\n"
            self.starDataText.setStyleSheet("QLabel { color : firebrick; }")
        if thisStar["near_cluster"]:
            new_info += "NEAR AN IDENTIFIED CLUSTER\n"
            self.starDataText.setStyleSheet("QLabel { color : red; }")
        if not (thisStar["near_star"] or thisStar["near_cluster"]):
            self.starDataText.setStyleSheet("QLabel { color : black; }")

        self.starDataText.setText(new_info)

        self.image.repaint()
        self.starDataText.repaint()

    def snapshot(self):
        thisStar = self.starData[self.idx]
        cen_x = thisStar["xcentroid"]
        cen_y = thisStar["ycentroid"]
        # get the subset of the data first
        # get the central pixel
        cen_x_pix = int(np.floor(cen_x))
        cen_y_pix = int(np.floor(cen_y))
        # we'll select a larger subset around that central pixel, then change the plot
        # limits to be just in the center, so that the object always appears at the
        # center
        buffer_half_width = int(np.ceil(self.plotWidth / 2) + 3)
        min_x_pix = cen_x_pix - buffer_half_width
        max_x_pix = cen_x_pix + buffer_half_width
        min_y_pix = cen_y_pix - buffer_half_width
        max_y_pix = cen_y_pix + buffer_half_width
        # then get this slice of the data
        snapshot_data = self.imageData[min_y_pix:max_y_pix, min_x_pix:max_x_pix]

        # When showing the plot I want the star to be in the very center. To do this I
        # need to get the values for the border in the new pixel coordinates
        cen_x_new = cen_x - min_x_pix
        cen_y_new = cen_y - min_y_pix
        # then make the plot limits
        min_x_plot = cen_x_new - 0.5 * self.plotWidth
        max_x_plot = cen_x_new + 0.5 * self.plotWidth
        min_y_plot = cen_y_new - 0.5 * self.plotWidth
        max_y_plot = cen_y_new + 0.5 * self.plotWidth

        fig, ax = bpl.subplots(figsize=[6, 5])
        vmax = np.max(snapshot_data)
        vmin = -5 * noise
        linthresh = max(0.01 * vmax, 5 * noise)
        norm = colors.SymLogNorm(vmin=vmin, vmax=vmax * 2, linthresh=linthresh)
        im = ax.imshow(snapshot_data, norm=norm, cmap=bpl.cm.lapaz)
        ax.set_limits(min_x_plot, max_x_plot, min_y_plot, max_y_plot)
        ax.scatter([cen_x_new], [cen_y_new], marker="x", c=bpl.almost_black, s=20)
        ax.remove_labels("both")
        ax.remove_spines(["all"])
        fig.colorbar(im, ax=ax)
        fig.savefig(temp_loc, dpi=100, bbox_inches="tight")
        plt.close(fig)


app = QApplication()

# The MainWindow class holds all the structure
window = MainWindow(stars_table, image_data, width)

# Execute application
app.exec_()

# The table will be modified as we go. We can then grab the rows selected and output
# the table
write_table = stars_table[np.where(stars_table["use_for_psf"])]
write_table.write(final_catalog, format="ascii.ecsv")
