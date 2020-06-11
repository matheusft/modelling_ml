from view import QtCore, QtWidgets
from os.path import join, abspath
import model as md
import sys
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas


# This class extends QtWidgets.QWidget. It is needed for plotting with matplotlib in PyQt5
# A QtWidgets.QWidget was manually promoted to a MplWidget parent class in the Qt Creator
class MplWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)

        self.canvas = FigureCanvas(Figure())

        vertical_layout = QtWidgets.QVBoxLayout()
        vertical_layout.addWidget(self.canvas)

        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)
        self.canvas.axes.axis('off') # Deactivate to show a white canvas in the initialisation


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller.

    Args:
        relative_path (str): path to the file.

    Returns:
        str: path that works for both debug and standalone app
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = abspath(".")

    return join(base_path, relative_path)


def configure_gui(ui, ml_model):
    _translate = QtCore.QCoreApplication.translate

    # Connecting load_file_pushButton - Dataset Load Tab
    ui.load_file_pushButton.clicked.connect(lambda ml_mode: load_dataset(ui, ml_model))

    # Connecting columnSelection_comboBox - Visualise Tab
    ui.columnSelection_comboBox.currentIndexChanged.connect(lambda: update_visualisation_options(ui, ml_model))

    # Connecting radio_button_change - Visualise Tab
    ui.boxplot_radioButton.clicked.connect(
        lambda: update_visualisation_widgets(ui, ml_model, ui.boxplot_radioButton.text()))
    ui.summary_radioButton.clicked.connect(
        lambda: update_visualisation_widgets(ui, ml_model, ui.summary_radioButton.text()))
    ui.plot_radioButton.clicked.connect(lambda: update_visualisation_widgets(ui, ml_model, ui.plot_radioButton.text()))
    ui.histogram_radioButton.clicked.connect(
        lambda: update_visualisation_widgets(ui, ml_model, ui.histogram_radioButton.text()))


def load_dataset(ui, ml_model):
    """Prompts the user to select an input file and call ml_model.read_dataset.

    Args:
        :param ml_model: An empty Machine Learning Model.
        :param ui:  The ui to be updated

    """
    # TODO: uncomment this when finish doing tests
    # fileDlg = QtWidgets.QFileDialog()
    # file_address = fileDlg.getOpenFileName()[0]
    # return_code = ml_model.read_dataset(file_address)
    file_address = '/Users/matheustorquato/Desktop/RSC_Data_2.xlsx'
    return_code = ml_model.read_dataset(file_address)

    if return_code == 0:
        populate_with_dataset_data(ui, ml_model.dataset)

    elif return_code == 1:  # Invalid file extension
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText("Error")
        msg.setInformativeText('Invalid Input format. \nTry Excel or .csv formats')
        msg.setWindowTitle("Error")
        msg.exec()

    elif return_code == 2:  # Exception while reading the file
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText("Error")
        msg.setInformativeText('Invalid Input File')
        msg.setWindowTitle("Error")
        msg.exec()


def populate_with_dataset_data(ui, dataset):
    # Fill dataset_tableWidget from the Dataset Load Tab with the head of the dataset
    ui.dataset_tableWidget.setRowCount(len(dataset.head(10)) + 1)  # +1 to add the Column Names in line 0
    ui.dataset_tableWidget.setColumnCount(len(dataset.columns))

    header = ui.dataset_tableWidget.horizontalHeader()  # uses this header in order to adjust the column width

    # Adding the labels at the top of the Table
    for i in range(ui.dataset_tableWidget.columnCount()):
        header.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)  # Column width fits the content
        qt_item = QtWidgets.QTableWidgetItem(dataset.columns[i])  # Creates an qt item
        qt_item.setTextAlignment(QtCore.Qt.AlignHCenter)  # Aligns the item in the horizontal center
        ui.dataset_tableWidget.setItem(0, i, qt_item)

    # Filling the Table with the dataset
    for i in range(ui.dataset_tableWidget.rowCount()):
        for j in range(ui.dataset_tableWidget.columnCount()):
            dataset_value = dataset.iloc[i, j]  # Get the value from the dataset
            # If the value is numeric, format it to two decimals
            dataset_value_converted = dataset_value if (type(dataset_value) is str) else '{:.2f}'.format(dataset_value)
            qt_item = QtWidgets.QTableWidgetItem(dataset_value_converted)  # Creates an qt item
            qt_item.setTextAlignment(QtCore.Qt.AlignHCenter)  # Aligns the item in the horizontal center
            ui.dataset_tableWidget.setItem(i + 1, j, qt_item)  # i+1 to skip the top row (column names)

    # Fill columnSelection_comboBox from the Visualise Tab
    for each_column in dataset.columns:
        ui.columnSelection_comboBox.addItem(each_column)


def update_visualisation_widgets(ui, ml_model, radio_name):
    selected_column = ui.columnSelection_comboBox.currentText()  # Get the selected value in the comboBox

    ui.columnSummary_textBrowser.clear()
    ui.columnSummary_textBrowser.append(ml_model.dataset[selected_column].describe().
                                        to_string(float_format='{:.2f}'.format).title().replace('     ','  = '))

    if ml_model.column_types_pd_series[selected_column].kind in 'iuf':  # iuf = i int (signed), u unsigned int, f float

        if radio_name == ui.plot_radioButton.text():
            print('Update Plot')
            plot_matplotlib_to_qt_widget(ml_model,[[0, 1, 2, 3, 4], [20, 50, 20, 15, 5]], ui.dataVisualisePlot_widget)
        elif radio_name == ui.boxplot_radioButton.text():
            print('Update Boxplot')
            plot_matplotlib_to_qt_widget(ml_model,[[0, 1, 2, 3, 4], [40, 10, 35, 30, 40]], ui.dataVisualisePlot_widget)
        elif radio_name == ui.histogram_radioButton.text():
            print('Update Histogram')
            # figure = ml_model.generate_histogram(selected_column)
            plot_matplotlib_to_qt_widget(ml_model,[[0, 1, 2, 3, 4], [10, 1, 20, 3, 40]],ui.dataVisualisePlot_widget)


def update_visualisation_options(ui, ml_model):
    """Update the available visualisation options for each column in the radio buttons.

    Args:
        :param ml_model: The Machine Learning Model.
        :param ui:  The ui to be updated

    """
    selected_column = ui.columnSelection_comboBox.currentText()  # Get the selected value in the comboBox
    # Create a list of all radioButton objects
    radio_buttons_list = [ui.summary_radioButton, ui.plot_radioButton, ui.boxplot_radioButton, ui.histogram_radioButton]
    # Check if the selected value in the columnSelection_comboBox is a numeric column in the dataset
    if ml_model.column_types_pd_series[selected_column].kind in 'iuf':  # iuf = i int (signed), u unsigned int, f float
        radio_buttons_list[0].setEnabled(True)
        radio_buttons_list[1].setEnabled(True)
        radio_buttons_list[2].setEnabled(True)
        radio_buttons_list[3].setEnabled(True)
        # Checking which radio button is checked
        radio_buttons_is_checked_list = [radio_buttons_list[0].isChecked(), radio_buttons_list[1].isChecked(),
                                         radio_buttons_list[2].isChecked(), radio_buttons_list[3].isChecked()]
        if any(radio_buttons_is_checked_list):
            # update the visualisation according to the current radio button
            update_visualisation_widgets(ui, ml_model,
                                         radio_buttons_list[radio_buttons_is_checked_list.index(1)].text())
    else:  # If not numeric, disable all visualisation options but the summary and checks it
        radio_buttons_list[0].setEnabled(True)
        radio_buttons_list[1].setEnabled(False)
        radio_buttons_list[2].setEnabled(False)
        radio_buttons_list[3].setEnabled(False)
        ui.summary_radioButton.setChecked(True)
        update_visualisation_widgets(ui, ml_model, ui.summary_radioButton.text())


def plot_matplotlib_to_qt_widget(ml_model,data,qt_widget):

    print('Plot updated')

    qt_widget.canvas.axes.axis('on')
    qt_widget.canvas.axes.clear()
    # qt_widget.canvas.axes.plot(data[0],data[1])
    # qt_widget.canvas.draw()

    # ml_model.dataset['Thickness'].plot
    import pandas as pd
    # pd.DataFrame(ml_model.dataset['Thickness']).hist(ax=qt_widget.canvas.axes)
    pd.DataFrame(ml_model.dataset['Thickness']).boxplot(ax=qt_widget.canvas.axes)
    qt_widget.canvas.draw()




try:
    sys._MEIPASS
    files_folder = resource_path('resources/')
except:
    pass
files_folder = resource_path('../resources/')
