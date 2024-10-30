import sys
import pandas as pd
import seaborn as sns
import mrcfile
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QGridLayout, 
                             QPushButton, QLineEdit, QFileDialog, QTableWidget, 
                             QTableWidgetItem, QLabel, QMessageBox, QHeaderView,
                             QScrollArea, QVBoxLayout)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np

import EMData


def get_clustered_data(df_particle,df_filament,selections):

    all_df = []
    all_sort_classes = []
    for i in range(len(selections)):
        selections_group = selections[i]
        selected_data = df_filament.iloc[selections_group]

        df_particle_group = pd.merge(df_particle, selected_data, left_on=['filename', 'helicaltube'],
                                             right_on=['filename', 'helicaltube'])

        class_indexes = np.array(df_particle_group['_rlnClassNumber'])
        unique_numbers, counts = np.unique(class_indexes, return_counts=True)
        percentages_np = (counts / len(class_indexes)) * 100
        number_percent_pairs = list(zip(unique_numbers, percentages_np))
        ranked_percentages_np = sorted(number_percent_pairs, key=lambda x: x[1], reverse=True)
        sort_classes = ranked_percentages_np

        all_df.append(df_particle_group)
        all_sort_classes.append(sort_classes)
    return all_df, all_sort_classes

def save_pdf_embedding(file_path, umap_2D, indexes, average_images, all_sort_classes):
    all_index = np.zeros(len(umap_2D))
    for i in range(len(indexes)):
        index = indexes[i]
        id = i+1
        all_index[index] = id
    with PdfPages(file_path) as pdf:
        plt.figure(figsize=(16, 16))
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, 3)
        ax4 = plt.subplot(2, 2, 4)

        sns.kdeplot(data=umap_2D, x='Dimension 1', y='Dimension 2', ax=ax1, fill=True, cmap='Blues')
        ax2.scatter(umap_2D['Dimension 1'],umap_2D['Dimension 2'],c='blue',alpha=0.1,s=5)

        print(len(set(all_index)))
        cmap = plt.cm.tab10
        colors = cmap(np.linspace(0, 1, len(set(all_index))))
        ax3.scatter(umap_2D['Dimension 1'],umap_2D['Dimension 2'],c=all_index,cmap=cmap,alpha=0.5,s=5)
        handles, labels = [], []
        for i, color in enumerate(colors):
            handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
            labels.append(f"cluster {i}")
        ax3.legend(handles, labels, title="Cluster index", loc='upper right')
        plt.tight_layout()
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        all_2D = average_images.data

        for i in range(len(all_sort_classes)):
            sorted_np = all_sort_classes[i]
            images_len = len(sorted_np)
            rows = int(np.ceil(images_len / 5))
            print(rows)
            if rows == 1:
                rows = rows+1
            fig, axs = plt.subplots(rows, 5, figsize=(10, int(2 * rows)))
            for j in range(images_len):
                row = j // 5
                col = j % 5
                id = sorted_np[j]
                axs[row, col].imshow(all_2D[int(id[0]) - 1], cmap='gray')
                axs[row, col].set_title(id[0] + ' : ' + str(round(float(id[1]), 3))+'%')
            fig.suptitle(f'The plot for the cluster {i + 1}')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()

class KDECanvas(FigureCanvas):
    def __init__(self, parent=None, data=None, selection_changed_callback=None):
        fig, self.ax = plt.subplots(1,2,figsize=(8, 4), dpi=100)
        super(KDECanvas, self).__init__(fig)
        self.setParent(parent)
        self.data=data
        self.ax[0].clear()  # Start with an empty plot
        self.selection_changed_callback = selection_changed_callback
        self.lasso = LassoSelector(self.ax[0], onselect=self.onselect)
        self.selected_indices = []
        self.selections = []

    def onselect(self, verts):
        path = Path(verts)
        self.selected_indices = [i for i, (x, y) in enumerate(zip(self.data['Dimension 1'], self.data['Dimension 2'])) if path.contains_point((x, y))]
        if self.selection_changed_callback:
            self.selection_changed_callback(self.selected_indices)

        self.selections = [selection for selection in
                           (self.remove_overlap(selection, self.selected_indices) for selection in self.selections) if
                           selection]

        if self.selected_indices:
            self.selections.append(self.selected_indices)
        self.highlight_selected_points()

    def remove_overlap(self, selection, new_selection):
        """Remove overlapping indices from a selection."""
        return [idx for idx in selection if idx not in new_selection]

    def highlight_selected_points(self):
        # Clear previous scatter plots but keep initial gray points
        self.ax[1].clear()
        self.ax[1].scatter(self.data['Dimension 1'], self.data['Dimension 2'], color='gray', s=1)  # Reset with default color

        #print(self.selections)
        # Assign a unique color to each selection
        for i, selected_indices in enumerate(self.selections):
            selected_x = self.data['Dimension 1'][selected_indices]
            selected_y = self.data['Dimension 2'][selected_indices]
            # Use a colormap to generate unique colors
            color = plt.cm.get_cmap('tab20b')(i / len(self.selections))
            self.ax[1].scatter(selected_x, selected_y, color=color, s=1)

        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data = pd.DataFrame()
        self.particle_data = pd.DataFrame()
        self.filament_table = pd.DataFrame({'filename':[],'helicaltube':[]})
        self.output_particle_data = pd.DataFrame()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Initialize a grid layout
        self.grid_layout = QGridLayout(self.central_widget)

        # Widgets for dimension reduction file loading
        self.DR_path_edit = QLineEdit()  # To display the selected file path
        self.DR_path_edit.returnPressed.connect(self.load_numpy_file)
        self.load_DR_button = QPushButton("Load dimension reduction file")
        self.load_DR_button.clicked.connect(self.load_numpy_file)

        # Add file loading widgets to the top row of the grid layout
        self.grid_layout.addWidget(self.DR_path_edit, 0, 0)
        self.grid_layout.addWidget(self.load_DR_button, 0, 1)

        # Widgets for star file loading
        self.star_path_edit = QLineEdit()  # To display the selected file path
        self.star_path_edit.returnPressed.connect(self.load_star_file)
        self.load_star_button = QPushButton("Load star file")
        self.load_star_button.clicked.connect(self.load_star_file)

        # Add file loading widgets to the top row of the grid layout
        self.grid_layout.addWidget(self.star_path_edit, 1, 0)
        self.grid_layout.addWidget(self.load_star_button, 1, 1)

        # Widgets for class average file loading
        self.average_path_edit = QLineEdit()  # To display the selected file path
        self.average_path_edit.returnPressed.connect(self.load_class_average_file)
        self.load_average_button = QPushButton("Load 2D class average file")
        self.load_average_button.clicked.connect(self.load_class_average_file)

        # Add file loading widgets to the top row of the grid layout
        self.grid_layout.addWidget(self.average_path_edit, 2, 0)
        self.grid_layout.addWidget(self.load_average_button, 2, 1)

        # add widgets for showing the major 2D class average
        self.image_scroll_area = QScrollArea()
        self.image_scroll_widget = QWidget()
        self.image_grid_layout = QGridLayout(self.image_scroll_widget)  # Use QGridLayout
        self.image_scroll_area.setWidgetResizable(True)
        self.image_scroll_area.setWidget(self.image_scroll_widget)

        self.grid_layout.addWidget(self.image_scroll_area, 3, 0)


        # Initialize KDE plot with Lasso selection
        self.kde_canvas = KDECanvas(self.central_widget, data=self.data, selection_changed_callback=self.update_data_table)
        self.grid_layout.addWidget(self.kde_canvas, 3, 1)

        # Initialize data table
        self.data_table = QTableWidget()
        self.grid_layout.addWidget(self.data_table, 4, 0)

        # Export button
        self.export_button = QPushButton("Export Selected Data")
        self.export_button.clicked.connect(self.export_data)
        self.grid_layout.addWidget(self.export_button, 5, 0)

        # Adjust the size of the main window
        self.setGeometry(100, 100, 1200, 600)

    def load_numpy_file(self):
        if self.DR_path_edit.text():
            file_name_numpy = self.DR_path_edit.text()
            print(file_name_numpy)
        else:
            file_name_numpy, _ = QFileDialog.getOpenFileName(self, "Dimension reduction file", "", "Numpy file (*.npy)")
        if file_name_numpy:
            self.DR_path_edit.setText(file_name_numpy)
            DM_data = np.load(file_name_numpy)
            self.data = pd.DataFrame({'Dimension 1':DM_data[:,0],'Dimension 2':DM_data[:,1]})
            self.refresh_plots_and_tables()
            self.kde_canvas.data = self.data

    def load_star_file(self):
        if self.star_path_edit.text():
            file_name_star = self.star_path_edit.text()
        else:
            file_name_star, _ = QFileDialog.getOpenFileName(self, "2D class meta files", "", "Star file (*.star)")
        if file_name_star:
            self.star_path_edit.setText(file_name_star)
            self.particle_data_class = EMData.read_data_df(file_name_star)
            self.particle_data = self.particle_data_class.star2dataframe()
            self.helical_data = self.particle_data_class.extract_helical_select(self.particle_data)
            filament_list = self.helical_data[1]
            for (filename,helicaltube) in filament_list:
                self.filament_table = pd.concat([self.filament_table,pd.DataFrame({'filename':[filename],'helicaltube':[helicaltube]})],ignore_index=True)
            self.data = self.data.iloc[:len(self.filament_table)]
            self.kde_canvas.data = self.data
            self.refresh_plots_and_tables()

    def load_class_average_file(self):
        if self.average_path_edit.text():
            file_name_average = self.average_path_edit.text()
        else:
            file_name_average, _ = QFileDialog.getOpenFileName(self, "2D class average files", "", "mrcs file (*.mrcs)")
        if file_name_average:
            self.average_path_edit.setText(file_name_average)
            self.average_image = mrcfile.mmap(file_name_average,mode='r')

    def refresh_plots_and_tables(self):
        if self.filament_table.empty or ('filename' not in self.filament_table.columns or 'helicaltube' not in self.filament_table.columns):
            self.kde_canvas.ax[0].clear()
            self.data_table.clear()
            self.data_table.setRowCount(0)
            self.data_table.setColumnCount(0)
            self.kde_canvas.draw()
        else:
            self.kde_canvas.ax[1].clear()
            sns.kdeplot(data=self.data, x='Dimension 1', y='Dimension 2', ax=self.kde_canvas.ax[0],fill=True, cmap='Blues')
            self.kde_canvas.ax[1].scatter(self.data['Dimension 1'], self.data['Dimension 2'], color='gray', s=1)
            self.kde_canvas.draw()

            # Clear previous data from table
            self.data_table.clear()
            self.data_table.setRowCount(len(self.filament_table))
            self.data_table.setColumnCount(len(self.filament_table.columns))
            self.data_table.setHorizontalHeaderLabels(self.filament_table.columns)

            for row in range(len(self.filament_table)):
                for col in range(len(self.filament_table.columns)):
                    item = QTableWidgetItem(str(self.filament_table.iloc[row, col]))
                    self.data_table.setItem(row, col, item)

    def update_data_table(self, selected_indices):
        selected_data = self.filament_table.iloc[selected_indices]
        self.data_table.setRowCount(len(selected_data))
        self.data_table.setColumnCount(len(selected_data.columns))
        self.data_table.setHorizontalHeaderLabels(selected_data.columns)
        for row, (idx, rowData) in enumerate(selected_data.iterrows()):
            for col, value in enumerate(rowData):
                item = QTableWidgetItem(str(value))
                self.data_table.setItem(row, col, item)

        # Resize columns to fit content
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        self.output_particle_data = pd.merge(self.particle_data, selected_data, left_on=['filename', 'helicaltube'], right_on=['filename', 'helicaltube'])
        #print(len(self.output_particle_data))

        class_indexes = np.array(self.output_particle_data['_rlnClassNumber'])
        unique_numbers, counts = np.unique(class_indexes, return_counts=True)
        percentages_np = (counts / len(class_indexes)) * 100
        number_percent_pairs = list(zip(unique_numbers, percentages_np))
        ranked_percentages_np = sorted(number_percent_pairs, key=lambda x: x[1], reverse=True)
        self.top_5_class = ranked_percentages_np[:6]
        self.sort_classes = ranked_percentages_np
        print(self.top_5_class)

        # Clear existing images and labels
        for i in reversed(range(self.image_grid_layout.count())):
            widget = self.image_grid_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        # Display corresponding images for selected indices in a three-column layout
        for i, (idx,percent) in enumerate(self.top_5_class):
            row, col = divmod(i, 3)  # Arrange in three columns

            image_array = self.get_image_array_for_index(idx)
            # Create a Matplotlib figure and add the image
            fig = plt.figure(figsize=(1.6, 1.6))  # Approx 128x128 pixels
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.imshow(image_array, cmap='gray', aspect='auto')  # Customize as needed
            ax.axis('off')  # Hide axes

            self.image_grid_layout.addWidget(canvas, row * 2, col)  # Add canvas to grid layout

            # Label for index
            index_label = QLabel(f'class {idx}: {round(percent,2)}%')
            index_label.setAlignment(Qt.AlignCenter)
            self.image_grid_layout.addWidget(index_label, row * 2 + 1, col)  # Add index label

    def get_image_array_for_index(self,idx):
        return self.average_image.data[int(idx)-1]

    def export_data(self):
        if not self.kde_canvas.selected_indices:
            QMessageBox.information(self, "No Data", "No data selected for export.")

        file_name, _ = QFileDialog.getSaveFileName(self, "Save the result", "", "pdf Files (*.pdf)")
        if file_name:
            file_name_star = file_name+'.star'
            file_name_pdf = file_name+ '.pdf'

            all_df, all_sort_classes = get_clustered_data(self.particle_data, self.filament_table, self.kde_canvas.selections)
            save_pdf_embedding(file_name_pdf, self.kde_canvas.data,
                               self.kde_canvas.selections, self.average_image, all_sort_classes)

            optics = self.particle_data_class.extractoptic()

            # write the clusters star file
            for i in range(len(all_df)):
                selected_particle_data = all_df[i]
                dataframe_out = selected_particle_data.drop(columns=['pid', 'filename', 'helicaltube', 'phi0', 'label'])
                metadata = list(dataframe_out.columns)
                metadata = [name for name in metadata]
                data_out = dataframe_out.values
                output = EMData.output_star(file_name_star, i, data_out, metadata)
                output.opticgroup(optics)
                output.writecluster()
            QMessageBox.information(self, "Export Successful", f"Data exported to {file_name}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
