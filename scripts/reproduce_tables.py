"""Create tables from evaluated outputs or copy the reference tables."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmtt.cli import tables

if __name__ == '__main__':
    tables()
