import matplotlib.pyplot as plt


class DynamicPlot:
    def __init__(self, memory_depth,
                 ):
        self.depth = memory_depth
        self.x1 = [i for i in range(memory_depth)]
        self.y1 = [0] * memory_depth
        self.x2 = [i for i in range(memory_depth)]
        self.y2 = [0] * memory_depth
        self.x3 = [i for i in range(memory_depth)]
        self.y3 = [0] * memory_depth
        self.x4 = [i for i in range(memory_depth)]
        self.y4 = [0] * memory_depth
        plt.ion()  # interactive mode on
        # switch to subplot 1
        plt.subplot(2, 2, 1)
        self.line1, = plt.plot(self.x1, self.y1)  # plot in subplot 1
        plt.axis([0, memory_depth, 0, 50])
        self.ax1 = plt.gca()  # get most of the figure elements
        plt.title("SUM")
        # switch to subplot 2
        plt.subplot(2, 2, 2)
        self.line2, = plt.plot(self.x2, self.y2)  # plot in subplot 1
        plt.axis([0, memory_depth, -1, 1])
        self.ax2 = plt.gca()  # get most of the figure elements
        plt.title("STATUS")
        # switch to subplot 3
        plt.subplot(2, 2, 3)
        self.line3, = plt.plot(self.x3, self.y3)  # plot in subplot 3
        plt.axis([0, memory_depth, -100, 100])
        plt.title("X-AXIS")
        self.ax3 = plt.gca()  # get most of the figure elements
        # switch to subplot 4
        plt.subplot(2, 2, 4)
        self.line4, = plt.plot(self.x4, self.y4)  # plot in subplot 4
        plt.axis([0, memory_depth, -100, 100])
        self.ax4 = plt.gca()  # get most of the figure elements
        plt.title("Y-AXIS")

    def add_data(self, val1, val2, val3, val4):
        self.y1 = self.y1[1:]+[val1]
        self.y2 = self.y2[1:]+[val2]
        self.y3 = self.y3[1:]+[val3]
        self.y4 = self.y4[1:]+[val4]

        self._redraw()

    def _redraw(self):
        self.line1.set_xdata(self.x1)
        self.line1.set_ydata(self.y1)  # set the curve with new data
        self.line2.set_xdata(self.x2)
        self.line2.set_ydata(self.y2)  # set the curve with new data
        self.line3.set_xdata(self.x3)
        self.line3.set_ydata(self.y3)  # set the curve with new data
        self.line4.set_xdata(self.x4)
        self.line4.set_ydata(self.y4)  # set the curve with new data
        '''
        self.ax3.relim()  # renew the data limits
        self.ax3.autoscale_view(True, True, True)  # rescale plot view
        self.ax4.relim()  # renew the data limits
        self.ax4.autoscale_view(True, True, True)  # rescale plot view
        '''
        plt.draw()  # plot new figure
        plt.pause(1e-17)  # pause to show the figure
