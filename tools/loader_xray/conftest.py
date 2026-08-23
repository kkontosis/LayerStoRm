"""Make the loader_xray modules importable by bare name from the test, regardless
of the pytest invocation cwd."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
