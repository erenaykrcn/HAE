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


"""from qiskit.circuit import ParameterVector
from qiskit.visualization import circuit_drawer

circuit_drawer(circuit=pqc1(ParameterVector('x', 4), ParameterVector('θ', N_PARAMS[1])), output="mpl").savefig("circuit_images/pqc1.png", dpi=300, transparent=True)
circuit_drawer(circuit=pqc2(ParameterVector('x', 4), ParameterVector('θ', N_PARAMS[2])), output="mpl").savefig("circuit_images/pqc2.png", dpi=300, transparent=True)
circuit_drawer(circuit=pqc3(ParameterVector('x', 4), ParameterVector('θ', N_PARAMS[3])), output="mpl").savefig("circuit_images/pqc3.png", dpi=300, transparent=True)
circuit_drawer(circuit=pqc4(ParameterVector('x', 4), ParameterVector('θ', N_PARAMS[4])), output="mpl").savefig("circuit_images/pqc4.png", dpi=300, transparent=True)
circuit_drawer(circuit=pqc5(ParameterVector('x', 4), ParameterVector('θ', N_PARAMS[5])), output="mpl").savefig("circuit_images/pqc5.png", dpi=300, transparent=True)
circuit_drawer(circuit=pqc6(ParameterVector('x', 4), ParameterVector('θ', N_PARAMS[6])), output="mpl").savefig("circuit_images/pqc6.png", dpi=300, transparent=True)
circuit_drawer(circuit=pqc7(ParameterVector('x', 4), ParameterVector('θ', N_PARAMS[7])), output="mpl").savefig("circuit_images/pqc7.png", dpi=300, transparent=True)
circuit_drawer(circuit=pqc8(ParameterVector('x', 4), ParameterVector('θ', N_PARAMS[8])), output="mpl").savefig("circuit_images/pqc8.png", dpi=300, transparent=True)"""