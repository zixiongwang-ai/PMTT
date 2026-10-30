"""Prepare ScanNet-1500 from official local files."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pmtt.cli import prepare

if __name__ == '__main__':
    prepare('scannet')
