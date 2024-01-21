from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

import sys
import pandas as pd
import seaborn as sns
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QTableWidget, QTableWidgetItem 
from PyQt5.QtWidgets import QHeaderView, QPushButton, QGridLayout, QFileDialog, QLineEdit, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

# Sample DataFrame
data = pd.DataFrame({
    'x': np.random.rand(1,2000)[0],
    'y': np.random.rand(1,2000)[0],
})

class KDECanvas(FigureCanvas):
    def __init__(self, parent=None, selection_changed_callback=None):
        fig, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        super(KDECanvas, self).__init__(fig)
        self.setParent(parent)
        sns.kdeplot(data=data, x='x', y='y', ax=self.ax)

        self.selection_changed_callback = selection_changed_callback
        self.lasso = LassoSelector(self.ax, onselect=self.onselect)
        self.selected_indices = []
        
    def onselect(self, verts):
        path = Path(verts)
        self.selected_indices = [i for i, (x, y) in enumerate(zip(data['x'], data['y'])) if path.contains_point((x, y))]
        
        if self.selection_changed_callback:
            self.selection_changed_callback(self.selected_indices)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Initialize a grid layout
        self.grid_layout = QGridLayout(self.central_widget)

        # Widgets for file loading
        self.file_path_edit = QLineEdit()  # To display the selected file path
        self.load_button = QPushButton("Load CSV File")
        self.load_button.clicked.connect(self.load_csv_file)

        # Add them to the top row of the grid layout
        self.grid_layout.addWidget(QLabel("CSV File:"), 0, 0)
        self.grid_layout.addWidget(self.file_path_edit, 0, 1)
        self.grid_layout.addWidget(self.load_button, 0, 2)

        # KDE plot with lasso tool
        self.kde_canvas = KDECanvas(self.central_widget, selection_changed_callback=self.update_data_table)
        self.grid_layout.addWidget(self.kde_canvas, 1, 1)  # Adjusted to row 1

        # Data display panel
        self.data_table = QTableWidget()
        self.grid_layout.addWidget(self.data_table, 1, 0)  # Adjusted to row 1

        # Export button
        self.export_button = QPushButton("Export Selected Data")
        self.export_button.clicked.connect(self.export_data)
        self.grid_layout.addWidget(self.export_button, 2, 0)  # Adjusted to row 2

        # Adjust the size of the main window
        self.setGeometry(100, 100, 1200, 600)


    def update_data_table(self, selected_indices):
        selected_data = data.iloc[selected_indices]
        self.data_table.setRowCount(len(selected_data))
        self.data_table.setColumnCount(len(selected_data.columns))
        self.data_table.setHorizontalHeaderLabels(selected_data.columns)

        for row, (idx, rowData) in enumerate(selected_data.iterrows()):
            for col, value in enumerate(rowData):
                item = QTableWidgetItem(str(value))
                self.data_table.setItem(row, col, item)

        # Resize columns to fit content
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    
    
    def export_data(self):
        # Get selected data from the DataFrame
        selected_indices = self.kde_canvas.selected_indices
        selected_data = data.iloc[selected_indices]

        # Export selected data
        # For example, to a CSV file
        selected_data.to_csv('selected_data.csv', index=False)
        print("Data exported successfully!")  # Or show a message to the user
    
    def load_csv_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_name:
            self.file_path_edit.setText(file_name)
            # Load and process the CSV file here
            # For example: self.data = pd.read_csv(file_name)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())

