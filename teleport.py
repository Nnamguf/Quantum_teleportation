#import packages
import qutip as qu
from qutip_qip.circuit import QubitCircuit, CircuitSimulator, Gate
from qutip_qip.device import Processor
from qutip_qip.noise import RelaxationNoise
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
def quantumteleport(alpha,beta,parties, noise = "", prob = 0.0, epsilon = 0.0, distance =0.0): #parties can only be two
    state = alpha * qu.fock(2, 0) + beta * qu.fock(2, 1)
    for i in range(parties - 1):
        # creates the state we want to teleport
        if noise == "loss":
            if np.random.rand() < 1-np.exp(-prob*(distance/(parties-1))):
                tpstates = qu.Qobj(np.zeros((2, 1)))
            else:
                tpstates = state
        else:
            tpstates = state
        # Creates a bellstate 1/sqrt(2) * (|00> +|11>)
        bell = qu.bell_state('00')
        # creates the statevector that we will send into the quantum teleportation
        zero_state = qu.tensor(tpstates, bell)
        #The amount of error that the gates have
        #If gateerror not active will there be no gateerror
        error = 0
        if noise == "gateerror":
            error = epsilon
        #define the quantum circuit for quantum teleportation
        q = QubitCircuit(3, num_cbits=2, reverse_states=False)
        # add the cnot gate on qubit 1 with control on the 0'th qubit
        q.add_gate("RX", targets=[0], arg_value=error)
        q.add_gate("CNOT", controls=[0], targets=[1])
        q.add_gate("RX", targets=[0], arg_value=-error)
        # add the Hadamard gate on qubit 0
        q.add_gate("RX", targets=[0], arg_value=error)
        q.add_gate("H", targets=[0])
        q.add_gate("RX", targets=[0], arg_value=-error)
        # Do alice measurement on qubit 0 and 1 and save the measurement as classical bits
        q.add_measurement("M0", targets=[0], classical_store=0)
        q.add_measurement("M1", targets=[1], classical_store=1)
        # adds noise if true
        if noise == "bit-flip":
            # does is as a probabilistic gate
            if np.random.rand() < prob:
                q.add_gate("X", targets=[2])
        # Do the correction depending on Alices measirements
        q.add_gate("RX", targets=[2], classical_controls=[1], arg_value=np.pi + error)
        q.add_gate("RZ", targets=[2], classical_controls=[0], arg_value=np.pi + error)
        # Define the CircuitSimulator that runs the QuantumCircuit
        sim = CircuitSimulator(q)
        # Define the CircuitSimulator that runs the QuantumCircuit
        # run the CircuitSimulator on the statevector
        try:
            result = sim.run(zero_state)
            # get out the final states
            res = result.get_final_states(0)
            #reads the state that bob sees
            amplitudes = ket_reader(res)
            state = amplitudes[0]*qu.fock(2,0) + amplitudes[1]*qu.fock(2,1)
        except:
            return "The states was lost during transmission. \nNo teleportation happened. \nTry again"
    return state
for i in range(10):
    tp = quantumteleport(0, -1, 2, noise ="", prob=0.2, distance = 1)
    print(tp)








