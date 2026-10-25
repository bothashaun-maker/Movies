"""Execute the generated model-setup cell, including its real Python subprocess.

Fixtures contain only the four required quantization descriptors, not model weights.
"""
import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/'minimax_h3_kaggle_complete.ipynb'


def write_quant_fixture(path, valid=True):
    header={};payload=bytearray()
    for suffix in ('attn.qkv_proj','attn.out_proj','mlp.fc1','mlp.fc2'):
        blob=json.dumps({'format':'int8_tensorwise','convrot':valid,'convrot_groupsize':256}).encode()
        header['blocks.0.'+suffix+'.comfy_quant']={
            'dtype':'U8','shape':[len(blob)],'data_offsets':[len(payload),len(payload)+len(blob)]}
        payload.extend(blob)
    encoded=json.dumps(header).encode()
    encoded+=b' '*((-len(encoded))%8)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(struct.pack('<Q',len(encoded))+encoded+payload)


class GeneratedSetup(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.nb=json.loads(NOTEBOOK.read_text())
        self.cells={cell['id']:''.join(cell['source']) for cell in self.nb['cells'] if cell['cell_type']=='code'}

    def test_all_embedded_programs_compile_in_generated_notebook(self):
        programs={'probe','kernel_probe','metadata_probe','script','code','compat_source','checkpoint_source'}
        found=set()
        for cell_id,source in self.cells.items():
            for node in ast.walk(ast.parse(source)):
                if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
                    for target in node.targets:
                        if isinstance(target,ast.Name) and target.id in programs:
                            compile(node.value.value,f'{cell_id}:{target.id}','exec')
                            found.add(target.id)
        self.assertEqual(found,programs)

    def prepare_scope(self, valid):
        if importlib.util.find_spec('safetensors') is None or importlib.util.find_spec('torch') is None:
            self.skipTest('safetensors and CPU torch are required to execute the real metadata reader.')
        scope={}
        source=self.cells['settings'].replace("Path('/kaggle/working/minimax_h3_dual_t4')",f'Path({str(self.root)!r})')
        exec(source,scope);exec(self.cells['catalog'],scope)
        source_file=self.root/'inputs/diffusion_models/fixture.safetensors'
        write_quant_fixture(source_file,valid)
        asset={'repo':'fixture/local','revision':'fixture','file':'diffusion_models/fixture.safetensors',
               'size':source_file.stat().st_size,'sha256':hashlib.sha256(source_file.read_bytes()).hexdigest()}
        mgpu=scope['COMFY']/'custom_nodes/ComfyUI-MultiGPU-Kaggle'
        mgpu.mkdir(parents=True)
        shutil.copy2(ROOT/'patches/h3_quant_compat.py',mgpu/'h3_quant_compat.py')
        scope.update(ASSETS=[asset],MODEL_DATASET=str(self.root/'inputs'),MODEL_STORAGE=str(self.root/'cache'),
                     STACK_READY=True,PY=Path(sys.executable),MGPU=mgpu)
        self.results=[]
        def run(args,**kwargs):
            self.assertFalse(scope['MODELS_READY'],'Rendering enabled before metadata validation completed')
            result=subprocess.run([str(arg) for arg in args],env=kwargs.get('env',scope['child_env']),
                                  capture_output=True,text=True)
            self.results.append(result)
            result.check_returncode()
            print(result.stdout,end='')
            return result
        scope['run']=run
        return scope,source_file

    def test_generated_cell_runs_real_metadata_subprocess_and_reuses_weights(self):
        scope,source_file=self.prepare_scope(valid=True)
        before=source_file.stat().st_mtime_ns
        for attempt in range(2):
            with self.subTest(attempt=attempt):
                output=io.StringIO()
                with redirect_stdout(output):exec(self.cells['weights'],scope)
                self.assertTrue(scope['MODELS_READY'])
                self.assertIn('H3 INT8 ConvRot metadata validated: 4 layers',output.getvalue())
                self.assertIn('All weights and metadata verified',output.getvalue())
        self.assertEqual(len(self.results),2,'Unexpected download or extra subprocess')
        self.assertEqual(source_file.stat().st_mtime_ns,before)
        target=scope['asset_target'](scope['ASSETS'][0])
        self.assertTrue(target.is_symlink());self.assertEqual(target.resolve(),source_file)

    def test_failed_metadata_leaves_rendering_disabled_and_retains_files(self):
        scope,source_file=self.prepare_scope(valid=False)
        output=io.StringIO()
        with redirect_stdout(output),self.assertRaises(subprocess.CalledProcessError):
            exec(self.cells['weights'],scope)
        self.assertFalse(scope['MODELS_READY'])
        self.assertTrue(source_file.is_file())
        self.assertIn('H3 tensor parallel requires INT8 ConvRot',self.results[0].stderr)
        self.assertNotIn('rendering is enabled',output.getvalue())


if __name__=='__main__':unittest.main()
