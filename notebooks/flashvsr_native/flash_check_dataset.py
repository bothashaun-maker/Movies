"""Import the actual CPU uploader dependencies before any model transfer."""
import importlib.metadata
import inspect
import subprocess
import sys


def check_runtime():
    import six
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kaggle.models.upload_file import UploadFile
    # Exercise the generated model that exposed the undeclared six dependency.
    assert UploadFile(token='dependency-probe').to_dict()['token'] == 'dependency-probe'
    upload = inspect.signature(KaggleApi.upload_files).parameters
    create = inspect.signature(KaggleApi.dataset_create_new).parameters
    if not {'folder', 'request'} <= set(upload):
        raise RuntimeError('Kaggle native uploader signature differs from the reviewed version.')
    if not {'folder', 'public', 'convert_to_csv', 'dir_mode'} <= set(create):
        raise RuntimeError('Kaggle private Dataset creation signature differs from the reviewed version.')
    versions = {}
    for name in ('kaggle', 'kagglesdk', 'six'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not installed separately'
    print('PASS: Dataset runtime imports in the notebook Python:', versions, flush=True)
    return KaggleApi


def prepare_runtime():
    """Match H3 Next: keep working notebook packages; install only when missing."""
    try:
        return check_runtime()
    except ModuleNotFoundError as error:
        missing = (error.name or '').split('.')[0]
        if missing == 'six':
            packages = ['six==1.17.0']
        elif missing in ('kaggle', 'kagglesdk'):
            packages = ['kaggle', 'six==1.17.0']
        else:
            raise
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *packages], check=True, timeout=180)
        return check_runtime()


if __name__ == '__main__':
    prepare_runtime()
