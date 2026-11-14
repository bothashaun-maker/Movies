"""Real Kaggle SDK path with only remote operations stubbed. Run in the CPU-only lock env."""
from contextlib import nullcontext
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import socket
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]/'notebooks/wan_gp_motion'
sys.path.insert(0, str(SOURCE))


@unittest.skipUnless(importlib.util.find_spec('kaggle'), 'Install the CPU requirements in a fresh isolated environment.')
class NativeSdkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        # Dummy credentials only: no user credentials or network access in this test.
        self.env = patch.dict(os.environ, {'KAGGLE_API_TOKEN': '', 'KAGGLE_USERNAME': 'jjp-sdk-test',
                                          'KAGGLE_KEY': 'not-a-real-api-key',
                                          'KAGGLE_CONFIG_DIR': str(self.root/'fake-config')})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.network = patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden in SDK test'))
        self.network.start()
        self.addCleanup(self.network.stop)
        from check_dataset_runtime import check_runtime
        self.api_class = check_runtime()

    def test_real_generated_upload_model_and_isolated_import(self):
        import six
        from kaggle.models.upload_file import UploadFile
        self.assertEqual(six.__version__, '1.17.0')
        self.assertEqual(UploadFile(token='test').to_dict()['token'], 'test')
        self.assertTrue(Path(six.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()))

    def test_working_notebook_packages_are_reused_without_install(self):
        from check_dataset_runtime import prepare_runtime
        with patch('check_dataset_runtime.subprocess.run', side_effect=AssertionError('Do not reinstall working packages')):
            self.assertIs(prepare_runtime(), self.api_class)

    def exercise_creation(self, reject=False):
        from dataset import create_dataset
        from requests import HTTPError, Response
        from kagglesdk.datasets.types.dataset_api_service import ApiCreateDatasetResponse
        api = self.api_class()
        api.authenticate()  # Actual legacy credential parsing; no network required.
        source = self.root/'input'
        source.mkdir()
        (source/'a.safetensors').write_bytes(b'input preserved')
        values = {'a.safetensors': b'input preserved', 'b.gguf': b'downloaded fixture'}
        assets = [dict(name=name, file=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(),
                       source='https://invalid.example/'+name) for name, content in values.items()]
        uploaded, submitted = {}, []

        def status(_):
            response = Response()
            response.status_code = 403  # Native SDK tolerates an unavailable status endpoint.
            raise HTTPError('stub status endpoint', response=response)

        def blob(path, quiet, blob_type, upload_context, content_type=None):
            path = Path(path)
            uploaded[path.name] = path.read_bytes()
            return 'test-token-'+path.name

        def submit(request):
            submitted.append(request)
            self.assertTrue(request.is_private)
            self.assertEqual(len(request.files), 3)
            self.assertEqual({f.token for f in request.files}, {'test-token-a.safetensors', 'test-token-b.gguf', 'test-token-manifest.json'})
            result = ApiCreateDatasetResponse()
            result.status = 'error' if reject else 'success'
            result.error = 'stub server rejection' if reject else ''
            return result

        def download(record, path):
            Path(path).write_bytes(values[record['name']])
            return Path(path)

        client = types.SimpleNamespace(datasets=types.SimpleNamespace(dataset_api_client=types.SimpleNamespace(create_dataset=submit)))
        original_upload = api.upload_files
        with patch.object(api, 'dataset_status', side_effect=status), \
             patch.object(api, '_upload_blob', side_effect=blob), \
             patch.object(api, 'build_kaggle_client', side_effect=lambda: nullcontext(client)), \
             patch('dataset.transfer', side_effect=download):
            if reject:
                with self.assertRaisesRegex(RuntimeError, 'not confirmed'):
                    create_dataset({'assets': assets}, self.root/'out', source, self.root/'cache', api=api)
            else:
                create_dataset({'assets': assets}, self.root/'out', source, self.root/'cache', api=api)
        self.assertEqual(len(submitted), 1)
        self.assertEqual(uploaded['a.safetensors'], values['a.safetensors'])
        self.assertEqual(uploaded['b.gguf'], values['b.gguf'])
        self.assertEqual(json.loads(uploaded['manifest.json']), {'assets': assets})
        self.assertEqual((source/'a.safetensors').read_bytes(), values['a.safetensors'])
        self.assertFalse((self.root/'cache/b.gguf').exists())
        self.assertEqual(api.upload_files, original_upload)
        self.assertEqual((self.root/'out/dataset-upload.json').exists(), not reject)

    def test_native_private_create_and_real_file_token_conversion(self):
        self.exercise_creation()

    def test_native_server_rejection_never_records_success(self):
        self.exercise_creation(reject=True)


if __name__ == '__main__':
    unittest.main()
