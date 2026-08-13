#import packages
import qutip as qu
from qutip_qip.circuit import QubitCircuit, CircuitSimulator, Gate
import numpy as np
import matplotlib.pyplot as plt
#define X and Z gates
x = np.array([[0,1],[1,0]])
z = np.array([[1,0],[0,-1]])
pauli = np.dstack([z,x])
#Create function for quantum teleportation
def quantumteleport(alpha,beta,parties): #parties can only be two
    # creates the state we want to teleport
    tpstates = alpha*qu.fock(2,0) + beta*qu.fock(2,1)
    # Creates a bellstate 1/sqrt(2) * (|00> +|11>)
    bell = qu.bell_state('00')
    zero_state = qu.tensor(tpstates,bell)
    q = QubitCircuit(3,num_cbits=2, reverse_states=False)
    q.add_gate("CNOT", controls=[0], targets=[1])
    q.add_gate("H", targets=[0])
    q.add_measurement("M0", targets=[0], classical_store=0)
    q.add_measurement("M1", targets=[1], classical_store=1)
    q.add_gate("Z", targets=[2], classical_controls=[0])
    q.add_gate("X", targets=[2], classical_controls=[1])
    sim = CircuitSimulator(q)
    result = sim.run(zero_state)
    res = result.get_final_states(0)
    vals, vecs =res.ptrace(2).eigenstates()
    mul = np.array([-alpha,beta])
    ket = mul * vecs[1].full().reshape(2,)
    ket[ket == -0.0] = 0.0
    ket = qu.Qobj(ket)
    print(ket)
quantumteleport(0,-1,2)






