"""Bundle the whole app into ONE runnable file:   python research_pipeline/build_single_file.py  ->  healthcare_research.pyz
Run it anywhere with Python 3.10+ (stdlib only):  python healthcare_research.pyz        (opens the full app on :8765)
Other modes keep working:  python healthcare_research.pyz ask "..."  |  understand "..."  |  library overview  |  train q.txt
"""
import shutil
import sys
import tempfile
import zipapp
from pathlib import Path

here = Path(__file__).resolve().parent
out = Path(sys.argv[1]) if len(sys.argv) > 1 else here.parent / "healthcare_research.pyz"
with tempfile.TemporaryDirectory() as tmp:
    pkg = Path(tmp) / "research_pipeline"
    shutil.copytree(here, pkg, ignore=shutil.ignore_patterns("__pycache__", "tests", "*.db", "build_single_file.py", "*.pyz"))
    (Path(tmp) / "__main__.py").write_text("from research_pipeline.__main__ import main\nmain()\n")
    zipapp.create_archive(tmp, out, interpreter="/usr/bin/env python3", compressed=True)
print(f"built {out} ({out.stat().st_size // 1024} KB)")
