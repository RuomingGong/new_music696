from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class GenerateCM(QMainWindow):
    def __init__(self, parent=None):
        super(GenerateCM, self).__init__()
        self.CM = np.zeros((10,10))
        #######################################
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.mainHBOX_param_scene = QHBoxLayout()

        #CM view
        layout_plot = QHBoxLayout()
        self.loaded_plot = CMViewer(self)
        #self.loaded_plot.setMinimumHeight(200)
        self.loaded_plot.update()
        layout_plot.addWidget(self.loaded_plot)

        self.mainHBOX_param_scene.addLayout(layout_plot)
        self.centralWidget.setLayout(self.mainHBOX_param_scene)

class CMViewer(QGraphicsView):
    def __init__(self, parent=None):
        super(CMViewer, self).__init__(parent)
        self.parent=parent
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        #self.toolbar = NavigationToolbar(self.canvas, self)

        # self.canvas.setGeometry(0, 0, 1600, 500 )
        layout = QVBoxLayout()
        #layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.canvas.show()


    def update(self):
        self.connectivityMat = self.parent.CM
        self.figure.clear()
        self.axes=self.figure.add_subplot(1,1,1)
        im = self.axes.imshow(self.connectivityMat)
        self.figure.colorbar(im)
        #self.axes.axis('off')
        self.axes.set_title('Path Matrix')
        self.canvas.draw()
        self.canvas.show()

def main():
    app = QApplication(sys.argv)
    ex = GenerateCM(app )
    ex.show()
    sys.exit(app.exec_( ))


if __name__ == '__main__':
    main()
