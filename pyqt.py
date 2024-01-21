from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

import sys
import pandas as pd
import seaborn as sns
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

# Sample DataFrame
data = pd.DataFrame({
    'x': np.random.rand(1,2000)[0],
    'y': np.random.rand(1,2000)[0],
})

class KDECanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        super(KDECanvas, self).__init__(fig)
        self.setParent(parent)
        sns.kdeplot(data=data, x='x', y='y', ax=self.ax, fill=True, cmap='blue')

        self.lasso = LassoSelector(self.ax, onselect=self.onselect)
        self.selected_indices = []

    def onselect(self, verts):
        path = Path(verts)
        self.selected_indices = [i for i, (x, y) in enumerate(zip(data['x'], data['y'])) if path.contains_point((x, y))]
        self.highlight_selected_points()

    def highlight_selected_points(self):
        # Highlight or process the selected points
        for i in self.selected_indices:
            self.ax.plot(data['x'][i], data['y'][i], 'ro')  # example: highlight in red
        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.kde_canvas = KDECanvas(self.central_widget)
        self.layout.addWidget(self.kde_canvas)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())

