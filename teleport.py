#import packages
import qutip as qu
from qutip_qip.circuit import QubitCircuit, CircuitSimulator, Gate
from qutip_qip.device import Processor
from qutip_qip.noise import RelaxationNoise
import numpy as np
# Creates a way of reading what bob ses of alpha, beta
def ket_reader(state, positions):
    n = len(state.dims[0])
    print(n)
    bits = [format(i, f"0{n}b") for i in range(2 ** n)]

    bit = []

    for b in bits:
        value = ''.join(b[-(p + 1)] for p in positions)
        bit.append(int(value, 2))

    vector = state.full().flatten()
    kets = [np.array([]) for _ in range(2*len(positions))]
    for i, amplitude in enumerate(vector):
        kets[bit[i]]  = np.append(kets[bit[i]],amplitude)
    sums = [np.sum(array) for array in kets]
    return sums
#Create function for quantum teleportation
def quantumteleport(alpha,beta,parties, noise = "", prob = 0.0, epsilon = 0.0, distance =0.0): #parties can only be two
    # creates the state we want to teleport
    # Creates a bellstate 1/sqrt(2) * (|00> +|11>)
    if noise == "loss":
        if np.random.rand() < 1-np.exp(-prob*distance):
            tpstates = qu.Qobj(np.zeros((2, 1)))
        else:
            tpstates = alpha * qu.fock(2, 0) + beta * qu.fock(2, 1)
    else:
        tpstates = alpha * qu.fock(2, 0) + beta * qu.fock(2, 1)
    bell = qu.bell_state('00')
    zero_state = qu.tensor(tpstates, bell)
    error = 0
    if noise == "gateerror":
        error = epsilon
    #define the quantum circuit for quantum teleportation
    q = QubitCircuit(3, num_cbits=2, reverse_states=False)
    # add the cnot gate on qubit 1 with control on the 0'th qubit
    q.add_gate("RX", targets=[0], arg_value=error)
    q.add_gate("CNOT", controls=[0], targets=[1])
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
    q.add_gate("X", targets=[2], classical_controls=[1])
    q.add_gate("RX", targets=[2], arg_value=error)
    q.add_gate("Z", targets=[2], classical_controls=[0])
    q.add_gate("RZ", targets=[2], arg_value=error)
    # Define the CircuitSimulator that runs the QuantumCircuit
    sim = CircuitSimulator(q)
    # creates the statevector that we will send into the quantum teleportation
    # Define the CircuitSimulator that runs the QuantumCircuit
    # run the CircuitSimulator on the statevector
    try:
        result = sim.run(zero_state)
        # get out the final states
        res = result.get_final_states(0)
        #reads the state that bob sees
        amplitudes = ket_reader(res,[0])
        print(amplitudes)
        ket = amplitudes[0]*qu.fock(2,0) + amplitudes[1]*qu.fock(2,1)
        return ket
    except:
        return "The states was lost during transmission. \nNo teleportation happened. \nTry again"
tp = quantumteleport(1, 0, 2, noise ="", epsilon = 0.1)
print(tp)


def entanglementswap():
    entangleAB = qu.bell_state('00')
    entangleAC= qu.bell_state('00')
    entangleBD = qu.bell_state('00')
    system = qu.tensor(entangleAB,entangleAC,entangleBD)
    e = QubitCircuit(6, num_cbits=4, reverse_states=False)
    e.add_gate("CNOT", controls=[0], targets=[2])
    e.add_gate("CNOT", controls=[1], targets=[4])
    e.add_gate("H", targets=[0])
    e.add_gate("H", targets=[1])
    e.add_measurement("M0", targets=[0], classical_store=0)
    e.add_measurement("M1", targets=[2], classical_store=1)
    e.add_measurement("M0", targets=[1], classical_store=2)
    e.add_measurement("M1", targets=[4], classical_store=3)
    e.add_gate("X", targets=[3], classical_controls=[1])
    e.add_gate("Z", targets=[3], classical_controls=[0])
    e.add_gate("X", targets=[5], classical_controls=[3])
    e.add_gate("Z", targets=[5], classical_controls=[1])
    #e.draw()
    simE = CircuitSimulator(e)
    result = simE.run(system)
    # get out the final states
    res = result.get_final_states(0)
    print(res)
    print(res.dims)
    amplitudes = ket_reader(res, [2,0])
    print(amplitudes)
entanglementswap()




