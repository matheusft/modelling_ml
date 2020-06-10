import os

#Check whether we are running in a windows system
is_windows = os.name == 'nt'

if is_windows:
    separator_slash = '\\'
    separator_colon = ";"
    multi_line = '^\n'
else:
    separator_slash = '/'
    separator_colon = ":"
    multi_line = ''

executable_python_code = 'executable.py'

files_to_add = "../resources/pickle/*.pckl:../resources/pickle/".replace(':',separator_colon).replace('/',separator_slash)
command_to_add_files = '--add-data "{}"'.format(files_to_add)

files_to_add = "../resources/images/Tri-Wall-Logo_NO_BG_small.png:../resources/images/".\
    replace(':',separator_colon).replace('/',separator_slash)
command_to_add_files += ' {}--add-data "{}"'.format(multi_line,files_to_add)

cmd1 = 'pyi-makespec --onedir {} {}'.format(executable_python_code,command_to_add_files)

print(cmd1)

os.system(cmd1)


