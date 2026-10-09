"""wrencode's tests: stdlib unittest, run from the repository root with
`python -m unittest discover -s tests -t .`. The package under src/ is put on
the path here, so no install is needed."""

import pathlib
import sys

_SRC = str(pathlib.Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
