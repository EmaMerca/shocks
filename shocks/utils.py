import os.path
from typing import List
__all__ = [
    "from_root",
    "handle_exception",
    "ROOT"
]


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def from_root(*args):
    return os.path.join(ROOT, *args)


def handle_exception(e, logger):
    logger.exception(e)
    exit(1)
