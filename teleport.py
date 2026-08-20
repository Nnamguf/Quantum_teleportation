#import packages
import qutip as qu
from qutip_qip.circuit import QubitCircuit, CircuitSimulator, Gate
from qutip_qip.device import Processor
from qutip_qip.noise import RelaxationNoise
from qutip.measurement import measure_observable
import numpy as np

# Creates a way of reading the state that is outputed from a qauntum circuit.
# Since qutip does not destroy the state when measured then there is need for somthing that can remove all the state
# that should have been gone. This is what ket_reader does.
# ket_reader takes two inputs the state you want to read and which qubits are relevant.
# And outputs a list of amplitudes for those qubits.
def ket_reader(state, positions):
    #Find the dimension of the hilbertspace ie. the amount of qubits in the hilbert space.
    n = len(state.dims[0])
    #Creates a list of all possible states in this hilbertspace
    bits = [format(i, f"0{n}b") for i in range(2 ** n)]
    #Creates a list which will contain the where each element in the state belongs
    # to in the output. So for a qubit bit will be two long so |0> and |1>.
    # And for a bell state bit will be 4 long so |00>, |01>, |10> and |11>
    bit = []
    for b in bits:
        value = ''.join(b[-(p + 1)] for p in positions)
        bit.append(int(value, 2))
    #take the state that have been teleported
    vector = state.full().flatten()
    #Take the amplitude of each element in the statevector and appends it to the right place
    # to create the amplitude of the output state and then returns that list.
    kets = [np.array([]) for _ in range(2*len(positions))]
    for i, amplitude in enumerate(vector):
        kets[bit[i]]  = np.append(kets[bit[i]],amplitude)
    sums = [np.sum(array) for array in kets]
    return sums


# quantumteleport is a function that teleports a qubit from Alice to Bob.
# quantumteleport take multiple inputs: Firstly it take the inputs alpha and beta
# sending the state psi = alpha |0> + beta |1>. Then it takes an input parties
# which is how many people there are who we send it through. So if parties is 3 then Alice send it to Bob and Bob
# sends the state to charlie. Then it take the input noise, which is what error do we want to simulate.
# There is 3 types of noise, "loss", "gateerror", "bit-flip". Any of there error types is activated by
# adding them to the list noise. prob and epsilon are inputs that tune the amount of error we see.
# Prob is used for "loss" and "bit-flip", and epsilon is used for "gateerror".
# If Fidelity is true then quantumteleport also outputs the fidelity between the send state and the teleported one.
# Seed is true then quantumteleport uses a seed which is 42. If Draw is true the quantum circuit will be drawn.
def quantumteleport(alpha,beta,parties = 2, noise = [], prob = 0.0, epsilon = 0.0,
                    distance =0.0, Fidelity = False,seed = False, Draw = False):
    #activates the seed if Seed = true
    if seed == True:
        rng = np.random.default_rng(42)
    else:
        rng = np.random.default_rng()
    # teleports through all parties.
    for i in range(parties - 1):
        # creates the state we want to teleport
        # Takes into account if there is loss in the system
        if "loss" in noise:
            #at random applies a loss to the state we want to teleport
            if rng.random() < 1-np.exp(-prob*(distance/(parties-1))):
                # If the state is lost then we output that the state was lost
                return "The states was lost during transmission. \nNo teleportation happened. \nTry again"
            else:
                tpstates = alpha * qu.fock(2, 0) + beta * qu.fock(2, 1)
        else:
            tpstates = alpha * qu.fock(2, 0) + beta * qu.fock(2, 1)
        # Creates a bellstate 1/sqrt(2) * (|00> +|11>). Which is the entanglement used to teleport.
        bell = qu.bell_state('00')
        #Combine the state we teleport with the bell state
        zero_state = qu.tensor(tpstates, bell)
        # Now the quantum circuit is created.
        # If gate error is true then every gate does a tiny error in the form of a RX rotation.
        if "gateerror" in noise:
            #The error is set to a random value from [-epsilon,epsilon[
            error = rng.uniform(epsilon, epsilon)
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
            # Do the correction depending on Alice's measurements
            # here the rotation is a tiny bit  larger or  smaller then expected creating error
            q.add_gate("RX", targets=[2], classical_controls=[1], arg_value=error+np.pi)
            q.add_gate("RZ", targets=[2], classical_controls=[0], arg_value=error+np.pi)
            # adds a bit-flip if that is true
            if "bit-flip" in noise:
                # Applies the gate in some procent of the time determined by prob.
                if rng.random.rand() < prob:
                    q.add_gate("X", targets=[2])
        # If there is no gate error we end up here
        else:
            # define the quantum circuit for quantum teleportation
            q = QubitCircuit(3, num_cbits=2, reverse_states=False)
            # add the cnot gate on qubit 1 with control on the 0'th qubit
            q.add_gate("CNOT", controls=[0], targets=[1])
            # add the Hadamard gate on qubit 0
            q.add_gate("H", targets=[0])
            # Do alice measurement on qubit 0 and 1 and save the measurement as classical bits
            q.add_measurement("M0", targets=[0], classical_store=0)
            q.add_measurement("M1", targets=[1], classical_store=1)
            # adds a bit-flip if that is true
            if "bit-flip" in noise:
                # does is as a probabilistic gate
                if rng.random.rand() < prob:
                    q.add_gate("X", targets=[2])
            # Do the correction depending on Alices measirements
            q.add_gate("X", targets=[2], classical_controls=[1])
            q.add_gate("Z", targets=[2], classical_controls=[0])
        # If draw is true the quantumcircuit made will be shown here
        if Draw == True:
            q.draw()
        # Define the CircuitSimulator that runs the QuantumCircuit
        sim = CircuitSimulator(q)
        # creates the statevector that we will send into the quantum teleportation
        # Define the CircuitSimulator that runs the QuantumCircuit
        # run the CircuitSimulator on the statevector
        # Try and except is for then there as been a loss of the teleported state.
        result = sim.run(zero_state)
        # get out the final states
        res = result.get_final_states(0)
        #reads the state that bob sees
        amplitudes = ket_reader(res,[0])
        #Creates the state
        ket = amplitudes[0]*qu.fock(2,0) + amplitudes[1]*qu.fock(2,1)
    #if fidelity is true the fidelity is calculated and return along the state
    if Fidelity == True:
        fidelity = np.abs(qu.fidelity(ket, tpstates)) ** 2
        return f"The teleported state is: {amplitudes[0]}|0> + {amplitudes[1]}|1>\n Fhe fidelity is: {fidelity}"
    else:
        return f"The teleported state is: {amplitudes[0]}|0> + {amplitudes[1]}|1>"


# Here we do entanglement swapping so teleporting a bell state from Alice and Bob to Charlie and Diane.
# The function entanglementswap takes multiple inputs.
# teleport is the state we want to teleport it can be "phi+", "phi-", "psi+", "psi-" which are the 4 bell states.
# It take in noise is only "gateerror" and is applies as noise = ["gateerror"].
# epsilon controls the streng of "gateerror". If seed it true then it uses the seed 42.
# If Draw is true then it show the quantum circuit. If Fidelity is true the output also contains the fidelity
# between bell state the Alice and Bob and the bell state Charlie and Diane shares in the end.
def entanglementswap(teleport = 'phi+', noise = [],epsilon = 0.01,seed =False,Draw = False, Fidelity = False):
    #Generates a seed if seed is true
    if seed == True:
        rng = np.random.default_rng(42)
    else:
        rng = np.random.default_rng()
    #Defines the possible bell state that can be teleported and maps them over to what qutip can read.
    mapping = {"phi+": '00', "phi-": '01', "psi+": '10', "psi-" : '11'}
    # pull out the bell state  we want to telepor
    teleport = mapping[teleport]
    # Create the bell state we will teleport
    entangleAB = qu.bell_state(teleport)
    # Create the entanglement needed for teleportation
    entangleAC= qu.bell_state('00')
    entangleBD = qu.bell_state('00')
    #defines the the full statevector
    system = qu.tensor(entangleAB,entangleAC,entangleBD)
    #create the quantum circuit
    # If gate error is true then every gate does a tiny error in the form of a RX rotation.
    if "gateerror" in noise:
        #defines the error form [-epsilon,epsilon[
        error = rng.uniform(epsilon, epsilon)
        # Creates a circuit that teleports Alice part of the bellstate we want to teleport to charlie
        # and teleports Bobs part to Diane. The circuit is two circuits that teleports a single qubit.
        e = QubitCircuit(6, num_cbits=4, reverse_states=False)
        # the error is created by over or underrotation the state when doing a gate.
        # this i done by applying an RX gate before each gate
        e.add_gate("RX", targets=[0], arg_value=error)
        e.add_gate("CNOT", controls=[0], targets=[2])
        e.add_gate("RX", targets=[1], arg_value=error)
        e.add_gate("CNOT", controls=[1], targets=[4])
        e.add_gate("RX", targets=[0], arg_value=error)
        e.add_gate("H", targets=[0])
        e.add_gate("RX", targets=[1], arg_value=error)
        e.add_gate("H", targets=[1])
        #measuse the qubits that Alice and Bob has access to.
        e.add_measurement("M0", targets=[0], classical_store=0)
        e.add_measurement("M1", targets=[2], classical_store=1)
        e.add_measurement("M0", targets=[1], classical_store=2)
        e.add_measurement("M1", targets=[4], classical_store=3)
        # applies a correction such that Chalie and Diane shares an entangled state.
        # There is a tiny over/under correction created by the rotation error+np.pi
        e.add_gate("RX", targets=[3], classical_controls=[1], arg_value=error+np.pi)
        e.add_gate("RZ", targets=[3],classical_controls=[0], arg_value=error+np.pi)
        e.add_gate("RX", targets=[5], classical_controls=[3], arg_value=error+np.pi)
        e.add_gate("RZ", targets=[5],classical_controls=[2], arg_value=error+np.pi)
    # If there is no gate error the gates a perfect gates no extra error.
    else:
        e = QubitCircuit(6, num_cbits=4, reverse_states=False)
        # Creates a circuit that teleports Alice part of the state we want to teleport to charlie
        # and teleports Bobs part to Diane
        e.add_gate("CNOT", controls=[0], targets=[2])
        e.add_gate("CNOT", controls=[1], targets=[4])
        e.add_gate("H", targets=[0])
        e.add_gate("H", targets=[1])
        # measuse the qubits that Alice and Bob has access to.
        e.add_measurement("M0", targets=[0], classical_store=0)
        e.add_measurement("M1", targets=[2], classical_store=1)
        e.add_measurement("M0", targets=[1], classical_store=2)
        e.add_measurement("M1", targets=[4], classical_store=3)
        # applies a correction such that Chalie and Diane shares an entangled state.
        e.add_gate("X", targets=[3], classical_controls=[1])
        e.add_gate("Z", targets=[3], classical_controls=[0])
        e.add_gate("X", targets=[5], classical_controls=[3])
        e.add_gate("Z", targets=[5], classical_controls=[2])
    # If draw is activated the quantum circuit will be shown
    if Draw == True:
        e.draw()
    #Now run the circuit
    simE = CircuitSimulator(e)
    result = simE.run(system)
    # get out the final states
    res = result.get_final_states(0)
    #uses the ketreader to get the the bellstate out again
    amplitudes = ket_reader(res, [2,0])
    #creates the bell state
    state = np.array(amplitudes)
    bellstate = qu.Qobj(state,dims=[[2, 2], [1]])
    # if fidelity is true then the output is both the entangled state and the fidelity
    # between then teleported bell state and the bell state that we wanted to teleport.
    if Fidelity == True:
        fidelity = np.abs(qu.fidelity(bellstate, entangleAB)) ** 2
        return f"The teleported entangled state is:\n {state[0]}|00> +{state[1]}|01>\n+ {state[2]}|10> + {state[3]}|11>\nThe fidelity is: {fidelity}"
    return f"The teleported entangled state is: {state[0]}|00> +{state[1]}|01> + {state[2]}|10> + {state[3]}|11> "



# quantumteleport_CV tries to teleport coherent states from alice to bob with continuous variables.
# Sadly the resolution of reach state in the fock space is to low meaning the fidelity will sometimes
# be very high (close to 1) and sometimes very low (0.01).
# The way to make it better is by increasing the squeezing parameter r
# and by making the fock space bigger by setting N to be higher. This is not possible since for high N
# we will run out of memory very fast.
# To see the physics look at part A in IV in this article https://arxiv.org/pdf/quant-ph/0604027
# quantumteleport_CV take 3 inputs and outputs the fidelity between the original state and the one bob has in the end.
# alpha is the coherent state we want to teleport, r is the squeezing parameter and N is the size of the fock space.
def quantumteleport_CV(alpha = 1,r=100, N = 10):
    #define state to teleport
    psi_input = qu.coherent(N, alpha)
    #define annihilation operators
    a = qu.destroy(N)
    I = qu.qeye(N)
    a0 = qu.tensor(a, I, I)  # input
    a1 = qu.tensor(I, a, I)  # Alice
    a2 = qu.tensor(I, I, a)  # Bob
    #Define 2 mode squeezing
    S2 = (r * (a1 * a2 - a1.dag() * a2.dag())).expm()
    #define full system state of the state we want to teleport and an EPR state.
    vac = qu.tensor(psi_input,qu.basis(N, 0), qu.basis(N, 0))
    state = S2 * vac
    #Define the beamsplitter acting on input and ALice
    theta = np.pi / 4
    U_bs = (theta * (a0.dag() * a1- a0 * a1.dag())).expm()
    #apply beamsplitter on the state
    BsState = U_bs * state
    # Now we do a homodyne detection on the output of the beamsplitter
    # Create the x and p operator after the beamsplitter
    x0 = (a0 + a0.dag()) / np.sqrt(2)
    p1 = (a1 - a1.dag()) / (1j * np.sqrt(2))
    #apply them on the state
    mx, state_after_x = measure_observable(BsState, x0)
    mp, state_after_p = measure_observable(BsState, p1)
    #Displace the state bob with the amount the ALice measured on the beamsplitter.
    D = qu.tensor(qu.qeye(N),qu.qeye(N),qu.displace(N, mx + mp*1j))
    output = D * BsState
    # Keep only Bob's mode
    bob_out = output.ptrace(2)
    #test if the teleportation is good.
    fidelity = np.abs(qu.fidelity(psi_input, bob_out)) ** 2
    return f"The fidelity is {fidelity}"


# quantumteleport_CV_gaussian is a better way of doing continuous variable teleportation.
# It uses a gaussian frame. So the work with a vector r with x and p's (pthe two qaudratures) and a covariance matrix.
# Any unitary operator does two things now first it is applied to the vector U r
# and is applied on the covariance matrix sigma like so U sigma U.T
# This allows us to do continuous variable without using fock space.
# To see the formalism ses https://arxiv.org/pdf/2102.05748
# quantumteleport_CV_gaussian takes 3 inputs:
# alpha is the coherent state we want to teleport
# r is the squeezing parameter
# fidelity is a parameter that tells if the output should contain the fidelity.
def quantumteleport_CV_gaussian(alpha,r=20, fidelity = False):
    # create the system
    # define the quadrature values as a vector
    x0 = np.sqrt(2) * np.real(alpha)
    p0 = np.sqrt(2) * np.imag(alpha)
    quad = np.array([
        x0,
        p0,
        0.0,
        0.0,
        0.0,
        0.0
    ])
    # define covariance matrix:
    sigma = 0.5 * np.eye(6)

    # create 2 mode squeezing
    F_TMS = np.eye(6)
    theta = 0
    S = np.array([
        [np.cosh(r), 0, -np.sinh(r) * np.cos(theta), -np.sinh(r) * np.sin(theta)],
        [0, np.cosh(r), -np.sinh(r) * np.sin(theta), np.sinh(r) * np.cos(theta)],
        [-np.sinh(r) * np.cos(theta), -np.sinh(r) * np.sin(theta), np.cosh(r), 0],
        [-np.sinh(r) * np.sin(theta), np.sinh(r) * np.cos(theta), 0, np.cosh(r)]
    ])
    F_TMS[2:6, 2:6] = S
    # create beamsplitter
    F_BS = np.eye(6)
    I = np.eye(2)
    eta = 1/np.sqrt(2)
    BS = np.block([[eta*I,eta*I],[-eta*I,eta*I]])
    F_BS[0:4, 0:4] = BS

    # Apply the two operations on the vector of quadratures and the covariance matrix
    quad = F_TMS @ quad
    quad = F_BS @ quad
    sigma = F_TMS @ sigma @ F_TMS.T
    sigma = F_BS @ sigma @ F_BS.T

    # Extract the quadratures from quad via a homodyne detection
    # Define measurement
    measured_indices = [0, 3]
    # define where to apply the dispalcement.
    output_indices = [4, 5]
    # Extract the measured quadratures in the vector quad
    quad_measured = quad[measured_indices]
    # Extract output quadratures  in the vector quad
    quad_output = quad[output_indices]

    # Extract from covariance:
    # We write the covariance as
    #
    #             measured       output
    #
    # measured      B              C^T
    #
    # output        C              A

    B = sigma[np.ix_(measured_indices,measured_indices)]
    A = sigma[np.ix_(output_indices, output_indices)]
    C = sigma[np.ix_(output_indices,measured_indices)]
    #Simulate the measurements
    # since there is some uncertency to the measurement of a quadrature we model that now
    measurement = np.random.multivariate_normal(quad_measured,B)
    # These are our actual experimental results what our model give for the homodyne detection
    u = measurement[0]
    v = measurement[1]
    #Setup the output on 2 for the given measurments on 0 and 1 of the system
    B_inv = np.linalg.inv(B)

    quad_conditional = (quad_output+ C @ B_inv @ (measurement - quad_measured))

    #displace the output state
    # created the displacement vector
    displacement = np.array([np.sqrt(2) * u,-np.sqrt(2) * v])
    # apply the displacemet
    quad_teleported = (quad_conditional+ displacement)
    #Create the teleported state
    teleportedstate =  1/np.sqrt(2)*(quad_teleported[0]+1j*quad_teleported[1])
    # if fidellity is true output the state and the fidelity.
    if fidelity == True:
        fidelity = np.abs(np.exp(-abs(teleportedstate - alpha)**2 / 2))**2
        return f"The state is: |{np.real(teleportedstate.item())} + {np.imag(teleportedstate.item())}j>\nThe fidelity is: {fidelity.item()}"
    else:
        return f"The state is: |{teleportedstate.item()}>"

# Here are some test function to use
#print(quantumteleport(0.6,0.8,parties = 2, noise = ["loss"], prob = 0.1, epsilon = 0.1,
                    distance =1, Fidelity = True,seed = False, Draw = False))

#print(entanglementswap(teleport = 'phi+', noise = ["gateerror"],epsilon = 0.1,seed =False,Draw = False, Fidelity = True))

#print(quantumteleport_CV(alpha = 1,r=100, N = 10))

#print(quantumteleport_CV_gaussian(1,r=20, fidelity = True))


