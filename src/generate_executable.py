import PyInstaller.__main__
import os

def get_project_root_directory():
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
    root_directory = path + '/../'
    return root_directory

root_directory = get_project_root_directory()
src_folder = root_directory + 'src/'
src_file = 'main.py'
icon_folder = root_directory + 'resources/images/'
data_folder = root_directory + 'data/'

PyInstaller.__main__.run([
    '--name={}'.format('Modelling ML'), # Generated exe file name
    ['--onedir', '--onefile'][0],  # Single directory or Single file
    '--windowed',
    '--add-data=%s' % os.path.join('resource', data_folder, '*.csv'),
    '--add-data=%s' % os.path.join('resource', data_folder, '*.xls'),
    '--add-data=%s' % os.path.join('resource', data_folder, '*.xlsx'),
    '--icon={}'.format(os.path.join('resource', icon_folder, 'brain.ico')),
    '--hidden-import={}'.format('scipy._lib.messagestream'),
    '--hidden-import={}'.format('scipy._lib.messagestream'),
    '--hidden-import={}'.format('pandas._libs.tslibs.timedeltas'),
    '--hidden-import={}'.format('sklearn.utils._cython_blas'),
    '--hidden-import={}'.format('sklearn.neighbors.quad_tree'),
    '--hidden-import={}'.format('sklearn.neighbors.typedefs'),
    '--hidden-import={}'.format('sklearn.tree._utils'),
    os.path.join(src_folder, src_file),
])

