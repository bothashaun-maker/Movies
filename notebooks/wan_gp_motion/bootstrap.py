"""Pinned isolated Python bootstrap, shared by the GPU studio and CPU Dataset maker."""
from pathlib import Path
import os
import sys
import zipfile
from common import bounded, transfer, valid

UV = dict(source='https://files.pythonhosted.org/packages/3a/a9/a83cee9b8cf63e57ce64ba27c77777cc66410e144fd178368f55af1fa18d/uv-0.8.22-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
          bytes=21252972, sha256='8efec4ef5acddc35f0867998c44e0b15fc4dace1e4c26d01443871a2fbb04bf6')


def ensure_environment(scratch, output, env):
    scratch, output = Path(scratch), Path(output)
    uv = scratch/'uv'
    if not uv.exists():
        bounded([sys.executable, Path(__file__), 'uv', scratch], output/'python-bootstrap.log', 180, env=env)
    python = scratch/'venv'/'bin'/'python'
    if not python.exists():
        bounded([uv, 'venv', '--python', '3.11.13', scratch/'venv'], output/'python-bootstrap.log', 300, env=env)
    return python, uv


if __name__ == '__main__':
    if sys.argv[1] == 'uv':
        scratch = Path(sys.argv[2])
        wheel = transfer(UV, scratch/'uv.whl', reserve=16*1024**2)
        with zipfile.ZipFile(wheel) as archive:
            name = next(name for name in archive.namelist() if name.endswith('.data/scripts/uv'))
            (scratch/'uv').write_bytes(archive.read(name))
        (scratch/'uv').chmod(0o700)
        wheel.unlink()
    elif sys.argv[1] == 'cloudflare':
        from launcher import CF
        transfer(CF, Path(sys.argv[2]), reserve=16*1024**2)
