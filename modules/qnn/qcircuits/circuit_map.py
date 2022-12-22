import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname, './'))

from pqc1 import pqc1
from pqc2 import pqc2
from pqc3 import pqc3

circuit_map = {
	1: pqc1,
	2: pqc2,
	3: pqc3,
}


N_PARAMS = {
	1: 4,
	2: 4,
	3: 8,	
}