# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

added_files = [
     ( 'resources\\pickle\\*.pckl',                         'resources\\pickle' ),
     ( 'resources\\images\\Tri-Wall-Logo_NO_BG_small.png',  'resources\\images' )
     ]

     #the source path here (1st path) is appended to the pathex path
     #The second path specifies the name of the folder to contain the files at run-time.
     #When accessing files in the code they must refer to the second path

a = Analysis(['src\\executable.py'], #executable file path
             pathex=['C:\\Users\\Matheus.Torquato\\Documents\\GitHub\\Tri_Wall'], #Root Path
             binaries=[],
             datas=added_files,
             hiddenimports=['scipy._lib.messagestream',
                            'scipy._lib.messagestream',
                            'pandas._libs.tslibs.timedeltas',
                            'sklearn.utils._cython_blas',
                            'sklearn.neighbors.quad_tree',
                            'sklearn.neighbors.typedefs',
                            'sklearn.tree._utils'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='Tri-Wall', #Executable file name
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False,
          icon='resources\\images\\tri_wall.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='executable_package',  #Executable directory name
               icon='resources\\images\\tri_wall.ico')
