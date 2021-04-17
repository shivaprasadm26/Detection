# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['StartScript.py'],
             pathex=['C:\\Users\\user\\AppData\\Roaming\\Python\\Python37\\site-packages\\PyQt5\\Qt\\bin', 'D:\\Projects\\Codebase'],
             binaries=[],
             datas=[('yolov2_tiny-face.h5', '.')],
             hiddenimports=['h5py', 'h5py.defs', 'h5py.utils', 'h5py.h5ac', 'h5py._proxy'],
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
          name='StartScript',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='StartScript')
