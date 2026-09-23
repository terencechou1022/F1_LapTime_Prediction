"""把專案根目錄加進 sys.path，讓 `f1lab` 不論在哪個工作目錄都找得到。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
