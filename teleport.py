#import packages
import qutip as qu
from qutip_qip.circuit import QubitCircuit, CircuitSimulator, Gate
import numpy as np
# Creates a way of reading what bob ses of alpha, beta
def ket_reader(state):
    vector = state.full().flatten()
    ket0 = np.array([])
    ket1 = np.array([])
    kets = [ket0,ket1]
    for i, amplitude in enumerate(vector):
        kets[i%2]  = np.append(kets[i%2],amplitude)
    kets = [np.sum(kets[0]),np.sum(kets[1])]
    return kets
#Create function for quantum teleportation
def quantumteleport(alpha,beta,parties): #parties can only be two
    # creates the state we want to teleport
    tpstates = alpha*qu.fock(2,0) + beta*qu.fock(2,1)
    # Creates a bellstate 1/sqrt(2) * (|00> +|11>)
    bell = qu.bell_state('00')
    #creates the statevector that we will send into the quantum teleportation
    zero_state = qu.tensor(tpstates,bell)
    #define the quantum circuit for quantum teleportation
    q = QubitCircuit(3,num_cbits=2, reverse_states=False)
    # add the cnot gate on qubit 1 with control on the 0'th qubit
    q.add_gate("CNOT", controls=[0], targets=[1])
    # add the Hadamard gate on qubit 0
    q.add_gate("H", targets=[0])
    # Do alice measurement on qubit 0 and 1 and save the measurement as classical bits
    q.add_measurement("M0", targets=[0], classical_store=0)
    q.add_measurement("M1", targets=[1], classical_store=1)
    # Do the correction depending on Alices measirements
    q.add_gate("X", targets=[2], classical_controls=[1])
    q.add_gate("Z", targets=[2], classical_controls=[0])
    # Define the CircuitSimulator that runs the QuantumCircuit
    sim = CircuitSimulator(q)
    # run the CircuitSimulator on the statevector
    result = sim.run(zero_state)
    # get out the final states
    res = result.get_final_states(0)
    #reads the state that bob sees
    amplitudes = ket_reader(res)
    ket = amplitudes[0]*qu.fock(2,0) + amplitudes[1]*qu.fock(2,1)
    return ket






