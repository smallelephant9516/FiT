from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
import numpy as np

# Sample DataFrame
data = pd.DataFrame({
    'x': np.random.rand(1, 20)[0],
    'y': np.random.rand(1, 20)[0],
})

class KDECanvas(FigureCanvas):
    def __init__(self, data, parent=None):
        fig, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        super(KDECanvas, self).__init__(fig)
        self.setParent(parent)
        self.data = data
        self.scatter = self.ax.scatter(self.data['x'], self.data['y'], color='gray')  # Initial scatter plot
        self.shift_pressed = False
        self.lasso = LassoSelector(self.ax, onselect=self.onselect)
        self.selections = []  # List to hold selections
        self.draw()

    def onselect(self, verts):
        path = Path(verts)
        new_selection = [i for i, (x, y) in enumerate(zip(self.data['x'], self.data['y'])) if path.contains_point((x, y))]

        # Remove overlaps from previous selections and filter out any that become empty
        self.selections = [selection for selection in (self.remove_overlap(selection, new_selection) for selection in self.selections) if selection]

        # Only add new selection if it's not empty
        if new_selection:
            self.selections.append(new_selection)
        self.highlight_selected_points()

    def remove_overlap(self, selection, new_selection):
        """Remove overlapping indices from a selection."""
        return [idx for idx in selection if idx not in new_selection]

    def highlight_selected_points(self):
        # Clear previous scatter plots but keep initial gray points
        self.ax.clear()
        self.ax.scatter(self.data['x'], self.data['y'], color='gray')  # Reset with default color

        print(self.selections)
        # Assign a unique color to each selection
        for i, selected_indices in enumerate(self.selections):
            selected_x = self.data['x'][selected_indices]
            selected_y = self.data['y'][selected_indices]
            # Use a colormap to generate unique colors
            color = plt.cm.get_cmap('tab20c')(i / len(self.selections))
            self.ax.scatter(selected_x, selected_y, color=color)

        self.draw()

class MainWindow(QMainWindow):
    def __init__(self, data):
        super().__init__()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.kde_canvas = KDECanvas(data, self.central_widget)
        self.layout.addWidget(self.kde_canvas)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = MainWindow(data=data)
    mainWin.show()
    sys.exit(app.exec_())
