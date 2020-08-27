# Modelling ML
 
 This is Modelling ML, a code-free Machine Learning (ML) tool presented in a Graphical user interface (GUI) 
 which covers the most important steps in the design of Classification and Regression Machine Learning models.
 <!--
 The idea behind this tool came initially from a personal need where I needed to quickly perform some pre-processing, 
 visualisation and training ML models from datasets in .csv and Microsoft Excel (.xls and .xlsx) format.
 -->
 Modelling ML is 100% coded in Python 3 and uses some of the most popular libraries for Data Science and ML such as 
 [Scikit-learn](https://scikit-learn.org/stable/), [Pandas](https://pandas.pydata.org/pandas-docs/stable/index.html#),
 [Matplotlib](https://matplotlib.org/), [Numpy](https://numpy.org/), etc. The GUI was built on top of 
 [Qt](https://www.qt.io/download) using PyQt5. 
 

Even tought 

Eventug it was tested is working fine needs more updates.
in the bottom of this documents you'll find....

#### For updating anything in the GUI 
1. Update the files from [resources/ui](https://github.com/matheusft/modelling_ml/tree/master/resources/ui) using 
[Qt Designer](https://build-system.fman.io/qt-designer-download). 
(Trust me, you don't want to build a GUI using code).
2. Run [src/convert_ui_to_py.py](https://github.com/matheusft/modelling_ml/tree/master/src/convert_ui_to_py.py) 
(Do not fiddle yourself with [src/view.py](https://github.com/matheusft/modelling_ml/tree/master/src/view.py) nor 
[src/ml_gui_resources_rc.py](https://github.com/matheusft/modelling_ml/tree/master/src/ml_gui_resources_rc.py). Let
the [pyuic5](https://pypi.org/project/pyqt5ac/) in [src/convert_ui_to_py.py](https://github.com/matheusft/modelling_ml/tree/master/src/convert_ui_to_py.py)
 do its jobs.)
3. Run **src/main.py**

Ideally, this tool should be self-explanatory...

> A user interface (UI) is like a joke. If you have to explain it, it’s not that good

In case this UI fails to achieve the same level as a nice joke, please, check below for some extra info regarding each 
separate tab from Modelling ML.

___

### Loading a Dataset:
<!--
<img src="https://github.com/matheusft/modelling_ml/blob/master/readme_page/Loading.gif?raw=true" alt="Kitten" title="A cute kitten" width="150" height="100" />
-->
<img src="https://github.com/matheusft/modelling_ml/blob/master/readme_page/Loading.gif?raw=true"/>

___

### Visualising the loaded Dataset:
<img src="https://github.com/matheusft/modelling_ml/blob/master/readme_page/Visualising.gif?raw=true"/>

___

### Pre-Processing the Dataset:
<img src="https://github.com/matheusft/modelling_ml/blob/master/readme_page/Pre_processing.gif?raw=true"/>

___

### Selecting the Machine Learning Algorithm:
<img src="https://github.com/matheusft/modelling_ml/blob/master/readme_page/Model_Sel.gif?raw=true"/>

___

### Selecting Input and Output Variables:
<img src="https://github.com/matheusft/modelling_ml/blob/master/readme_page/Input_Output.gif?raw=true"/>

___

### Training the Model:
<img src="https://github.com/matheusft/modelling_ml/blob/master/readme_page/Training.gif?raw=true"/>

#### To do List:
* Unordered list can use asterisks
* Unordered list can use asterisks
* Unordered list can use asterisks

