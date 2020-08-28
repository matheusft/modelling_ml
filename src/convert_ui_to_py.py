import os
import glob

full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)

root_directory = path + '/../'
ui_directory = root_directory+ 'resources/ui/'

ui_file_path = glob.glob("{}*.ui".format(ui_directory))[0]
resources_file_path = glob.glob("{}*.qrc".format(ui_directory))[0]

src_directory = root_directory+ 'src/'
ui_python_converted_file_path = src_directory+'view.py'
resources_python_converted_file_path = src_directory+'ml_gui_resources_rc.py'

cmd1 = 'pyuic5 {} -o {}'.format(ui_file_path, ui_python_converted_file_path)
os.system(cmd1)

cmd2 = 'pyrcc5 {} -o {}'.format(resources_file_path, resources_python_converted_file_path)
os.system(cmd2)



