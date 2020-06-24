import os

cmd1 = 'pyuic5 ../resources/ui/ml_gui.ui -o view.py'
os.system(cmd1)

cmd2 = 'pyrcc5 ../resources/ui/ml_gui_resources.qrc -o ml_gui_resources_rc.py'
os.system(cmd2)





