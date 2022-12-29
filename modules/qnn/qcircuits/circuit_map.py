import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname, './'))

from pqc1 import pqc1
from pqc2 import pqc2
from pqc3 import pqc3
from pqc4 import pqc4
from pqc5 import pqc5
from pqc6 import pqc6
from pqc7 import pqc7
from pqc8 import pqc8
from pqc9 import pqc9


circuit_map = {
	1: pqc1,
	2: pqc2,
	3: pqc3,
	4: pqc4,
	5: pqc5,
	6: pqc6,
	7: pqc7,
	8: pqc8,
	9: pqc9,
}


N_PARAMS = {
	1: 4,
	2: 4,
	3: 8,
	4: 8,
	5: 8,
	6: 8,
	7: 8,
	8: 16,
	9: 16,
}