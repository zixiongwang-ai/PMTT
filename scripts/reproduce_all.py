"""Run the complete PMTT paper protocol or full refit."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmtt.cli import all_tasks

if __name__ == '__main__':
    all_tasks()
