import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname, './'))

from pqc1 import pqc1
from pqc2 import pqc2

circuit_map = {
	1: pqc1,
	2: pqc2,
}