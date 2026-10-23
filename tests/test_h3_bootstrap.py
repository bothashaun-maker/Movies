"""Regression checks for Kaggle's missing-ensurepip and partial-venv failure.

Run: python -m unittest discover -s tests -v
Optional real download/recovery check: python tests/test_h3_bootstrap.py --integration
The integration check uses temporary disk and does not install CUDA or model weights.
"""
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib.request
import uuid
import zipfile
from unittest.mock import patch

NOTEBOOK=Path(__file__).resolve().parents[1]/'minimax_h3_kaggle_dual_t4.ipynb'


def bootstrap_scope(root):
    notebook=json.loads(NOTEBOOK.read_text())
    tree=ast.parse(''.join(notebook['cells'][4]['source']))
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef)]
    scope=dict(ROOT=root,ENV=root/'venv',PY=root/'venv/bin/python',
               child_env=os.environ.copy(),os=os,shutil=shutil,subprocess=subprocess,
               urllib=__import__('urllib.request'),hashlib=hashlib,uuid=uuid,Path=Path)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(NOTEBOOK),'exec'),scope)
    return scope


class BootstrapTests(unittest.TestCase):
    def test_notebook_compiles(self):
        notebook=json.loads(NOTEBOOK.read_text())
        for i,cell in enumerate(notebook['cells']):
            if cell['cell_type']=='code':compile(''.join(cell['source']),f'cell{i}','exec')

    def test_missing_pip_rebuilds_and_preserves_partial_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            scope=bootstrap_scope(Path(tmp))
            env=scope['ENV'];(env/'bin').mkdir(parents=True)
            scope['PY'].touch();(env/'keep-me').write_text('original')
            wheel=io.BytesIO()
            with zipfile.ZipFile(wheel,'w') as archive:
                archive.writestr('uv-0.9.26.data/scripts/uv',b'test executable')
            payload=wheel.getvalue()
            commands=[]
            def run(args,**kwargs):
                commands.append(list(map(str,args)))
                if 'venv' in args:
                    (env/'bin').mkdir(parents=True);scope['PY'].touch()
            scope['run']=run
            scope['child_env'].update(PYTHONPATH='/bad',PYTHONHOME='/bad',VIRTUAL_ENV='/bad')
            with patch.object(subprocess,'run',side_effect=[
                subprocess.CompletedProcess([],1,stderr='No module named pip'),
                subprocess.CompletedProcess([],0,stdout='3.12.12')]), \
                 patch.object(urllib.request,'urlopen',return_value=io.BytesIO(payload)), \
                 patch.object(hashlib,'sha256') as sha:
                sha.return_value.hexdigest.return_value='b7e89798bd3df7dcc4b2b4ac4e2fc11d6b3ff4fe7d764aa3012d664c635e2922'
                scope['ensure_environment']()
            backups=list(Path(tmp).glob('venv.incomplete-*'))
            self.assertEqual(len(backups),1)
            self.assertEqual((backups[0]/'keep-me').read_text(),'original')
            self.assertTrue(any('--seed' in cmd and '--managed-python' in cmd for cmd in commands))
            self.assertFalse(any('ensurepip' in arg for cmd in commands for arg in cmd))
            self.assertFalse({'PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'} & scope['child_env'].keys())

    def test_bad_checksum_never_executes_or_replaces_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            scope=bootstrap_scope(Path(tmp))
            scope['ENV'].mkdir();(scope['ENV']/'keep-me').touch()
            with patch.object(urllib.request,'urlopen',return_value=io.BytesIO(b'bad')), \
                 patch.object(subprocess,'run') as run:
                with self.assertRaisesRegex(AssertionError,'checksum'):scope['ensure_environment']()
            run.assert_not_called()
            self.assertTrue((scope['ENV']/'keep-me').exists())

    def test_healthy_environment_needs_no_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            scope=bootstrap_scope(Path(tmp))
            scope['PY'].parent.mkdir(parents=True);scope['PY'].touch()
            with patch.object(subprocess,'run',return_value=subprocess.CompletedProcess([],0)), \
                 patch.object(urllib.request,'urlopen') as download:
                scope['ensure_environment']()
            download.assert_not_called()


def integration():
    with tempfile.TemporaryDirectory(prefix='h3-bootstrap-') as tmp:
        scope=bootstrap_scope(Path(tmp))
        # Same state as a failed Debian/Kaggle venv: interpreter exists, pip absent.
        subprocess.run([sys.executable,'-m','venv','--without-pip',scope['ENV']],check=True)
        scope['ensure_environment']()
        scope['run']([scope['PY'],'-m','pip','--version'])
        # Reuse must work even if all subsequent network requests fail.
        with patch.object(urllib.request,'urlopen',side_effect=AssertionError('unexpected download')):
            scope['ensure_environment']()
        print('PASS: real pinned Python creation, pip, partial-venv recovery and offline reuse')


if __name__=='__main__':
    if '--integration' in sys.argv:integration()
    else:unittest.main()
