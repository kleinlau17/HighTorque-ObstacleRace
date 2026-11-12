"""Pinned numerical/image runtime shared by simulation and robot releases."""
import json
from pathlib import Path
import cv2
import numpy as np


def verify_dependencies():
    expected=json.loads(Path(__file__).with_name('dependencies.json').read_text())
    actual={'numpy':np.__version__,'opencv':cv2.__version__}
    for name,value in actual.items():
        if value!=expected[name]:
            raise ValueError('Navigation dependency differs: %s=%s, expected %s; use the shared navigation environment' % (name,value,expected[name]))
    return actual
