from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, ClassicalRegister, IBMQ
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import EfficientSU2
from qiskit.algorithms.optimizers import GradientDescent, ADAM
from scipy.optimize import minimize

import numpy as np
import time

def kernel(j, k):
    c = 1
    sig = 1
    return c*np.exp((-(j-k)**2)/(2*sig**2))


def exp_val(p, q, N, approx=True, N_shot=10):
    if approx:
        aev = 0
        meas = [i for i in range(N)]
        rng = np.random.default_rng()
        j = rng.choice(meas, size=N_shot, p=p)
        k = rng.choice(meas, size=N_shot, p=q)
        for l in range(N_shot):
            aev += kernel(j[l], k[l])
        return aev/N_shot
    else:
        ev = 0
        for j in range(N):
            for k in range(N):
                ev += kernel(j,k)*p[j]*q[k]
        return ev


def L_MMD(p, q, N, approx=True, N_shot=10):
    return exp_val(q, q, N, approx, N_shot) - 2*exp_val(q, p, N, approx, N_shot) + exp_val(p, p, N, approx, N_shot)


def fwht(p):
    a = p.copy()
    h = 1
    while h < len(a):
        for i in range(0, len(a), 2*h):
            for j in range(i, i+h):
                x = a[j]
                y = a[j+h]
                a[j] = x + y
                a[j+h] = x - y
        a = np.divide(a, np.sqrt(2))
        h *= 2
    return a


def prep_circ(reps, d, thetas, gates, measure=True, assign=True):
    ckt = EfficientSU2(d, su2_gates=gates, reps=reps, entanglement='linear', insert_barriers=False, skip_final_rotation_layer=True)
    
    if assign:
        num_params = len(gates)*d*reps

        params = ckt.ordered_parameters

        param_vals = {params[i]: thetas[i] for i in range(num_params)}

        ckt.assign_parameters(param_vals, inplace=True)
        if measure:
            ckt.measure_all()
        
    return ckt


def cost(thetas, target, gates, reps, d, N):
    ckt = prep_circ(reps, d, thetas, gates, False)
    prepared_state = Statevector.from_instruction(ckt).data    
    cst = 1 - np.square(np.abs(np.vdot(prepared_state, target)))
    
    return cst


def pqc_load(A_vals, reps=2, gates=['ry'], optimizer='BFGS'):
    N = len(A_vals)
    d = int(np.ceil(np.log2(N)))

    num_params = len(gates)*d*reps

    init_theta = [np.random.uniform(-np.pi, np.pi) for i in range(num_params)]

    st = time.time()
    res = minimize(cost, init_theta, method=optimizer, args=(A_vals, gates, reps, d, N), options={'maxiter' : 500})
    et = time.time()

    theta_0 = res.x
    ckt = prep_circ(reps, d, theta_0, gates, False)
    sv = Statevector.from_instruction(ckt)

    # print(ckt.decompose())

    # print(sv.data)

    return sv.data, cost(theta_0, A_vals, gates, reps, d, N), et-st, num_params

def init_load(A_vals):
    N = len(A_vals)
    d = int(np.ceil(np.log2(N)))

    ckt = QuantumCircuit(d)

    st = time.time()
    ckt.initialize(A_vals, qubits=range(d))
    et = time.time()

    sv = Statevector.from_instruction(ckt)

    return sv.data, 1 - np.square(np.abs(np.vdot(sv.data, A_vals))), et-st

# N = 4
# A_probs = list(np.random.random(N))

# diff = int(2**np.ceil(np.log2(N)) - N)
# for _ in range(diff):
#     A_probs.append(0)

# A_probs = list(np.divide(A_probs, sum(A_probs)))

# # print("Sum of probs : ", sum(A_probs))

# A_vals = list(np.sqrt(A_probs))

# print("Target : ", A_vals)

# pqc_load(A_vals, gates=['ry'])