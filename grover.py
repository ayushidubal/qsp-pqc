from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library.standard_gates import RYGate, MCXGate, ZGate
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.algorithms import Grover, AmplificationProblem
from qiskit.utils import QuantumInstance
from qiskit.visualization import plot_histogram

import numpy as np

from qiskit import IBMQ
IBMQ.load_account()

def get_aln(al, n):
    """ Returns the binary form upto n bits of precision of the input decimal data. """
    aln = ""
    for i in range(n):
        al *= 2
        aln += str(int(al))
        al = al - int(al)
        
    # aln = "0." + aln
    return aln

def check_get_aln(aln, al):
    aln = aln[2:]
    check_al = 0
    p = 0.5
    for dig in aln:
        check_al += p*int(dig)
        p /= 2
    print(abs(al - check_al))

def aln_to_gate(aln, n):
    """ Constructs the gate that corresponds to the input binary string - X for bit '1' and I for bit '0'. """
    qc = QuantumCircuit(n)
    for i, dig in enumerate(aln):
        if dig == '1':
            qc.x(i)
    return qc.to_gate(label=aln)    

def get_sv(sv, modified):
    vals = []
    
    if modified == False:
        print("  Out   Data  Flag  Amplitude    Probability")
        for j in range(len(sv.data)):
            if np.abs(sv.data[j]) > 0.00001:
                vals.append((bin(j)[2:].rjust(d+n+1, '0')[::-1][0:d], bin(j)[2:].rjust(d+n+1, '0')[::-1][d:d+n], bin(j)[2:].rjust(d+n+1, '0')[::-1][d+n:], "{:.8f}".format(np.abs(sv.data[j])), "{:.8f}".format(np.abs(sv.data[j])**2)))        
    else:
        print("  Out   Data   Ref  Flag  Amplitude           Probability")
        for j in range(len(sv.data)):
            if np.abs(sv.data[j]) > 0.00001:
                vals.append((bin(j)[2:].rjust(d+2*n+1, '0')[::-1][0:d], bin(j)[2:].rjust(d+2*n+1, '0')[::-1][d:d+n], bin(j)[2:].rjust(d+2*n+1, '0')[::-1][d+n:d+2*n], bin(j)[2:].rjust(d+2*n+1, '0')[::-1][d+2*n:], "{:.8f}".format(np.abs(sv.data[j])), "{:.8f}".format(np.abs(sv.data[j])**2)))        
    vals.sort(key = lambda x: x[0])
    return vals

def initializer(A, d, m, n, modified):
    """ Initializes a circuit with the appropriate registers for Grover's state preparation method, or the modified version. 
        Also puts the first register in an equal superposition. """
    out = QuantumRegister(d, name = "out")
    data = QuantumRegister(n, name = "data")
    ref = QuantumRegister(n, name = "ref")
    flag = QuantumRegister(1, name = "flag")
    
    if modified:
        circ = QuantumCircuit(out, data, ref, flag, name = "init")
    else:
        circ = QuantumCircuit(out, data, flag, name = "init")
        
    k = int(np.log2(m))
    circ.h(range(k))
        
    return circ

def oracle(A, d, m, n):
    """ Loads the data array onto the circuit, controlled by register out as index. """
    circ = QuantumCircuit(d+n, name = "oracle")
    for i in range(m):
        aln_gate = aln_to_gate(A[i], n).control(num_ctrl_qubits = d, label = str(A[i]), ctrl_state = bin(i)[2:].rjust(d, "0")[::-1])
        circ.append(aln_gate, [i for i in range(0, d+n)])
    return circ

def rotate(A_vals_n, A, d, m):
    """ Applies a CRY(θ_i) Gate with θ_i = ith data element (upto n bits), and controlled by out register as index. """
    circ = QuantumCircuit(d+1, name = "rot")
    for i in range(m):
        theta_i = np.pi - 2*np.arcsin(A_vals_n[i])
        ry_gate = RYGate(theta = theta_i)
        circ.append(ry_gate.control(num_ctrl_qubits = d, label = str(A[i]), ctrl_state = bin(i)[2:].rjust(d, "0")[::-1]), [j for j in range(d+1)]) 
    return circ

def phase_oracle(d, n, modified):
    """ Flips the phase of the state to be amplified. """
    out = QuantumRegister(d, name = "out")
    data = QuantumRegister(n, name = "data")
    ref = QuantumRegister(n, name = "ref")
    flag = QuantumRegister(1, name = "flag")
    
    if modified:
        # Good state has ref and flag registers as all 0s
        circ = QuantumCircuit(out, data, ref, flag, name = "Phase Oracle")
        circ.x(flag)
        circ.append(ZGate().control(num_ctrl_qubits = n, label = "phase", ctrl_state = "0"*n), [i for i in range(d+n, d+(2*n)+1)])
        circ.x(flag)
    else:
        # Good state has flag register as all 0s
        circ = QuantumCircuit(out, data, flag, name = "Phase Oracle")
        circ.x(flag)
        circ.z(flag)
        circ.x(flag)
    
    return circ

def load(A_vals, n):
    m = len(A_vals)
    d = int(np.ceil(np.log2(m)))

    if d - np.log2(m) != 0:
        m = 2**d

    print(f"Data : {A_vals}")
    A = [get_aln(A_vals[i], n) for i in range(m)]
    print(f"Binary form upto {n} places : {A}")
    dec = 2**n
    A_vals_n = [int(A[i], 2)/dec for i in range(m)]
    print(f"Decimal data after rounding to {n} places : {A_vals_n}")
    print(f"Probabilities after rounding to {n} places : {np.square(A_vals_n)}")

    qc_orig = initializer(A, d, m, n, False)

    black_box = oracle(A, d, m, n)
    qc_orig.append(black_box, qargs = [i for i in range(0,d+n)])

    rot = rotate(A_vals_n, A, d, m)
    qc_orig.append(rot, qargs = [i for i in range(0,d)] + [d+n])

    backend = Aer.get_backend('aer_simulator')
    quantum_instance = QuantumInstance(backend)

    good_states = [bin(i)[2:].rjust(2, '0') + A[i] + '0' for i in range(len(A))]
    problem = AmplificationProblem(oracle = phase_oracle(d, n, False), state_preparation = qc_orig, objective_qubits = [i for i in range(0,d+n+1)], is_good_state = good_states)
    grover = Grover(quantum_instance=quantum_instance)

    out = QuantumRegister(d, name = "out")
    data = QuantumRegister(n, name = "data")
    flag = QuantumRegister(1, name = "flag")
    check = ClassicalRegister(d+1, name = "check")

    state_prep = QuantumCircuit(out, data, flag, check, name = "State preparation")

    # Construct the Grover circuit and apply the Grover operator to amplify the good states
    
    grover_ckt = grover.construct_circuit(problem, power = int((np.pi/4)*m*d))
    state_prep.append(grover_ckt, [i for i in range(0, d+n+1)])

    # Unload the data from data register by reapplying the oracle
    state_prep.append(black_box, qargs = [i for i in range(0,d+n)])

    # Measure the flag register
    state_prep.measure([out[i] for i in range(d)] + [flag[0]], check)

    backend = Aer.get_backend('aer_simulator')

    res = execute(state_prep, backend, shots=1000).result()
    counts = res.get_counts()
    print("Counts:", counts)

    selected_counts = {}
    s = 0
    for k in counts.keys():
        # Store in dict if flag bit is 0
        if k[0] == '0':
            selected_counts[k[1:]] = counts[k]
            s += counts[k]
    
    for k in selected_counts:
        selected_counts[k] /= s

    s = 0
    A_dict = {}
    for i in range(m):
        A_dict[bin(i)[2:].rjust(d, '0')[::-1]] = (A_vals[i])**2
        s += (A_vals[i])**2
    
    for i in range(m):
        A_dict[bin(i)[2:].rjust(d, '0')[::-1]] /= s

    print("Selected counts: ", selected_counts)
    print("Expected: ", A_dict)

# n = 2
# m = 2
# A_vals = [0.5, 0.25]
# load(A_vals, n, m)