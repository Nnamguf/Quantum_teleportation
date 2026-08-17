#import packages
import qutip as qu
from qutip_qip.circuit import QubitCircuit, CircuitSimulator, Gate
from qutip_qip.device import Processor
from qutip_qip.noise import RelaxationNoise
from qutip.measurement import measure_observable
import numpy as np
import time
# Creates a way of reading what bob ses of alpha, beta
def ket_reader(state, positions):
    #Find the dimension of the hilbertspace
    n = len(state.dims[0])
    #Creates a list of all possible states in this hilbertspace
    bits = [format(i, f"0{n}b") for i in range(2 ** n)]
    #Creates a list which will contain the where each element in the state belongs to in the statevector we want to output
    bit = []
    for b in bits:
        value = ''.join(b[-(p + 1)] for p in positions)
        bit.append(int(value, 2))
    #take the state that have been teleported
    vector = state.full().flatten()
    #Take the amplitude of each element in the statevector and appends it to the right place to create the amplitude of
    #the output state
    kets = [np.array([]) for _ in range(2*len(positions))]
    for i, amplitude in enumerate(vector):
        kets[bit[i]]  = np.append(kets[bit[i]],amplitude)
    sums = [np.sum(array) for array in kets]
    return sums
#Create function for quantum teleportation
def quantumteleport(alpha,beta,parties, noise = [], prob = 0.0, epsilon = 0.0, distance =0.0,
                    Fidelity = False): #parties can only be two
    #teleports to all parties
    for i in range(parties - 1):
        # creates the state we want to teleport
        # Takes into account if there is loss in the system
        if "loss" in noise:
            #at random applies a loss to the state we want to teleport
            if np.random.rand() < 1-np.exp(-prob*(distance/(parties-1))):
                tpstates = qu.Qobj(np.zeros((2, 1)))
            else:
                tpstates = alpha * qu.fock(2, 0) + beta * qu.fock(2, 1)
        else:
            tpstates = alpha * qu.fock(2, 0) + beta * qu.fock(2, 1)
        # Creates a bellstate 1/sqrt(2) * (|00> +|11>)
        bell = qu.bell_state('00')
        #Combine the state we teleport with the bell state
        zero_state = qu.tensor(tpstates, bell)
        #Applies the gate error. If there is error stays 0 else it is replace by epsilon
        error = 0
        if "gateerror" in noise:
            error = epsilon
        #define the quantum circuit for quantum teleportation
        q = QubitCircuit(3, num_cbits=2, reverse_states=False)
        # add the cnot gate on qubit 1 with control on the 0'th qubit
        # The RX gate is to apply a tiny error
        q.add_gate("RX", targets=[0], arg_value=error)
        q.add_gate("CNOT", controls=[0], targets=[1])
        # add the Hadamard gate on qubit 0
        # The RX gate is to apply a tiny error
        q.add_gate("RX", targets=[0], arg_value=error)
        q.add_gate("H", targets=[0])
        # Do alice measurement on qubit 0 and 1 and save the measurement as classical bits
        q.add_measurement("M0", targets=[0], classical_store=0)
        q.add_measurement("M1", targets=[1], classical_store=1)
        # adds a bit-flip if that is true
        if "bit-flip" in noise:
            # does is as a probabilistic gate
            if np.random.rand() < prob:
                q.add_gate("X", targets=[2])
        # Do the correction depending on Alices measirements
        # The RX gate is to apply a tiny error
        q.add_gate("X", targets=[2], classical_controls=[1])
        q.add_gate("RX", targets=[2], arg_value=error)
        # The Rz gate is to apply a tiny error
        q.add_gate("Z", targets=[2], classical_controls=[0])
        q.add_gate("RZ", targets=[2], arg_value=error)
        # Define the CircuitSimulator that runs the QuantumCircuit
        sim = CircuitSimulator(q)
        # creates the statevector that we will send into the quantum teleportation
        # Define the CircuitSimulator that runs the QuantumCircuit
        # run the CircuitSimulator on the statevector
        # Try and except is for then there as been a loss of the teleported state.
        try:
            result = sim.run(zero_state)
            # get out the final states
            res = result.get_final_states(0)
            #reads the state that bob sees
            amplitudes = ket_reader(res)
            #Creates the state
            ket = amplitudes[0]*qu.fock(2,0) + amplitudes[1]*qu.fock(2,1)
        except:
            return "The states was lost during transmission. \nNo teleportation happened. \nTry again"
    if Fidelity == True:
        fidelity = np.abs(qu.fidelity(ket, tpstates)) ** 2
        return f"The teleported state is: {ket}\n Fhe fidelity is: {fidelity}"
    else:
        return f"The teleported state is: {ket}"

#Define how to do quantum teleportation
def entanglementswap(teleport = 'phi+'):
    #Defines the possible bell state that can be teleported and maps them over to what qutip can read.
    mapping = {"phi+": '00', "phi-": '01', "psi+": '10', "psi-" : '11'}
    # pull out the bell state  we want to telepor
    teleport = mapping[teleport]
    # Create the bell state we will teleport
    entangleAB = qu.bell_state(teleport)
    # Create the entanglement needed for teleprotation
    entangleAC= qu.bell_state('00')
    entangleBD = qu.bell_state('00')
    #defines the the full statevector
    system = qu.tensor(entangleAB,entangleAC,entangleBD)
    #defiens the quantum circuit
    e = QubitCircuit(6, num_cbits=4, reverse_states=False)
    #Creates a circuit that teleports Alice part of the state we want to teleport to charlie
    # and teleports Bobs part to Diane
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
    e.add_gate("Z", targets=[5], classical_controls=[2])
    #Now run the circuit
    simE = CircuitSimulator(e)
    result = simE.run(system)
    # get out the final states
    res = result.get_final_states(0)
    #uses the ketreader to get the the bellstate out again
    amplitudes = ket_reader(res, [2,0])
    #creates the bell state
    state = np.array(amplitudes)
    state = qu.Qobj(state,dims=[[2, 2], [1]])
    return f"The teleported entangled state is: {ket}"




def quantumteleport_CV(r, x_in,p_in,Ideal = True, N = 20):
    #define annihilation operators for EPR
    a = qu.tensor(qu.destroy(N), qu.qeye(N))
    b = qu.tensor(qu.qeye(N), qu.destroy(N))
    #Define 2 mode squeezing
    S2 = (r * (a * b - a.dag() * b.dag())).expm()
    #define EPR state
    vac = qu.tensor(qu.basis(N, 0), qu.basis(N, 0))
    epr = S2 * vac
    print(epr)
    #Define the coherent state we want to teleport
    tpin = qu.coherent(N, (x_in + p_in * 1j)/np.sqrt(2))
    # define the beamsplitter operator for balanced beamsplitter and phi = 0
    IN = qu.tensor(qu.destroy(N), qu.qeye(N),qu.qeye(N))
    A = qu.tensor(qu.qeye(N), qu.destroy(N), qu.qeye(N))
    B = qu.tensor(qu.qeye(N),qu.qeye(N), qu.destroy(N))
    theta = np.pi / 4
    U_bs = (theta * (IN.dag()*A - IN*A.dag())).expm()
    #apply beamsplitter on the state
    state = qu.tensor(tpin,epr)
    BsState = U_bs * state
    #Measure the x_- and p_+
    # first define the new output operators
    # Output-mode quadratures
    x_In = (IN + IN.dag()) / np.sqrt(2)
    x_A = (A + A.dag()) / np.sqrt(2)
    p_In = (IN - IN.dag()) / (1j * np.sqrt(2))
    p_A = (A - A.dag()) / (1j * np.sqrt(2))
    x_minus = U_bs.dag()*  (x_A-x_In) / np.sqrt(2) * U_bs
    p_plus = U_bs.dag() * (p_A + p_In) / np.sqrt(2) * U_bs

    #apply them on the state
    mean_xminus = qu.expect(x_minus, BsState)
    mean_pplus = qu.expect(p_plus, BsState)
    mx, state_after_x = measure_observable(BsState, x_minus)
    mp, state_after_p = measure_observable(state_after_x, p_plus)
    #Displace the state bob
    D = qu.tensor(qu.qeye(N),qu.qeye(N),qu.displace(N, mx - mp*1j))
    output = D * state_after_p
    # Keep only Bob's mode
    bob_out = output.ptrace(2)
    #test if close
    fidelity = np.abs(qu.fidelity(tpin, bob_out)) ** 2
    return fidelity
print(quantumteleport_CV(3,1,1, N = 10))

