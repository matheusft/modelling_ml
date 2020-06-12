# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '../resources/ui/ml_gui.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(767, 460)
        self.tabs_widget = QtWidgets.QTabWidget(Dialog)
        self.tabs_widget.setGeometry(QtCore.QRect(20, 20, 731, 391))
        self.tabs_widget.setObjectName("tabs_widget")
        self.dataset_load_tab = QtWidgets.QWidget()
        self.dataset_load_tab.setObjectName("dataset_load_tab")
        self.load_file_pushButton = QtWidgets.QPushButton(self.dataset_load_tab)
        self.load_file_pushButton.setGeometry(QtCore.QRect(220, 0, 113, 32))
        self.load_file_pushButton.setObjectName("load_file_pushButton")
        self.dataset_tableWidget = QtWidgets.QTableWidget(self.dataset_load_tab)
        self.dataset_tableWidget.setGeometry(QtCore.QRect(10, 40, 691, 311))
        self.dataset_tableWidget.setToolTip("")
        self.dataset_tableWidget.setToolTipDuration(30000)
        self.dataset_tableWidget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.dataset_tableWidget.setObjectName("dataset_tableWidget")
        self.dataset_tableWidget.setColumnCount(0)
        self.dataset_tableWidget.setRowCount(0)
        self.tabs_widget.addTab(self.dataset_load_tab, "")
        self.visualise_tab = QtWidgets.QWidget()
        self.visualise_tab.setObjectName("visualise_tab")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.visualise_tab)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 10, 171, 131))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.columnSelection_comboBox = QtWidgets.QComboBox(self.verticalLayoutWidget)
        self.columnSelection_comboBox.setObjectName("columnSelection_comboBox")
        self.verticalLayout.addWidget(self.columnSelection_comboBox)
        self.plot_radioButton = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.plot_radioButton.setObjectName("plot_radioButton")
        self.verticalLayout.addWidget(self.plot_radioButton)
        self.boxplot_radioButton = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.boxplot_radioButton.setObjectName("boxplot_radioButton")
        self.verticalLayout.addWidget(self.boxplot_radioButton)
        self.histogram_radioButton = QtWidgets.QRadioButton(self.verticalLayoutWidget)
        self.histogram_radioButton.setObjectName("histogram_radioButton")
        self.verticalLayout.addWidget(self.histogram_radioButton)
        self.columnSummary_textBrowser = QtWidgets.QTextBrowser(self.visualise_tab)
        self.columnSummary_textBrowser.setGeometry(QtCore.QRect(19, 160, 291, 181))
        self.columnSummary_textBrowser.setObjectName("columnSummary_textBrowser")
        self.dataVisualisePlot_widget = MplWidget(self.visualise_tab)
        self.dataVisualisePlot_widget.setGeometry(QtCore.QRect(339, 10, 320, 331))
        self.dataVisualisePlot_widget.setObjectName("dataVisualisePlot_widget")
        self.tabs_widget.addTab(self.visualise_tab, "")
        self.inputoutput_tab = QtWidgets.QWidget()
        self.inputoutput_tab.setObjectName("inputoutput_tab")
        self.tabs_widget.addTab(self.inputoutput_tab, "")
        self.pre_process_tab = QtWidgets.QWidget()
        self.pre_process_tab.setObjectName("pre_process_tab")
        self.tabs_widget.addTab(self.pre_process_tab, "")
        self.model_selection_tab = QtWidgets.QWidget()
        self.model_selection_tab.setObjectName("model_selection_tab")
        self.tabs_widget.addTab(self.model_selection_tab, "")
        self.train_tab = QtWidgets.QWidget()
        self.train_tab.setObjectName("train_tab")
        self.tabs_widget.addTab(self.train_tab, "")
        self.test_tab = QtWidgets.QWidget()
        self.test_tab.setObjectName("test_tab")
        self.tabs_widget.addTab(self.test_tab, "")

        self.retranslateUi(Dialog)
        self.tabs_widget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Machine Learning Generic Modeller"))
        self.load_file_pushButton.setText(_translate("Dialog", "Load File"))
        self.tabs_widget.setTabText(self.tabs_widget.indexOf(self.dataset_load_tab), _translate("Dialog", "Dataset Load"))
        self.plot_radioButton.setText(_translate("Dialog", "Plot"))
        self.boxplot_radioButton.setText(_translate("Dialog", "Boxplot"))
        self.histogram_radioButton.setText(_translate("Dialog", "Histogram"))
        self.tabs_widget.setTabText(self.tabs_widget.indexOf(self.visualise_tab), _translate("Dialog", "Visualise"))
        self.tabs_widget.setTabText(self.tabs_widget.indexOf(self.inputoutput_tab), _translate("Dialog", "Input/Output"))
        self.tabs_widget.setTabText(self.tabs_widget.indexOf(self.pre_process_tab), _translate("Dialog", "Pre-Process"))
        self.tabs_widget.setTabText(self.tabs_widget.indexOf(self.model_selection_tab), _translate("Dialog", "Model Selction"))
        self.tabs_widget.setTabText(self.tabs_widget.indexOf(self.train_tab), _translate("Dialog", "Train"))
        self.tabs_widget.setTabText(self.tabs_widget.indexOf(self.test_tab), _translate("Dialog", "Test"))

from controller import MplWidget
