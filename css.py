from qaravan.core import Circuit, H, CNOT
from qaravan.algebraQ import qaravan_to_stim
import stim
from typing import List, Tuple
import numpy as np
from collections import Counter

def css_stabilizers(hx: np.ndarray, hz: np.ndarray) -> List[stim.PauliString]:
    """ convert parity matrices to a list of CSS stabilizers """
    stabilizers = []
    for row in hx:
        pauli_string = stim.PauliString(''.join(['X' if bit == 1 else 'I' for bit in row]))
        stabilizers.append(pauli_string)
    
    for row in hz:
        pauli_string = stim.PauliString(''.join(['Z' if bit == 1 else 'I' for bit in row]))
        stabilizers.append(pauli_string)

    return stabilizers

def hamming_mat() -> np.ndarray:
    return np.array([
    [1, 0, 1, 0, 1, 0, 1], 
    [0, 1, 1, 0, 0, 1, 1], 
    [0, 0, 0, 1, 1, 1, 1]])

def steane_encoder() -> Circuit:
    """ returns the Steane code encoder circuit """
    return Circuit([H(6), H(5), H(4), 
                CNOT([2,0]), 
                CNOT([1,0]), 
                CNOT([3,6]), 
                CNOT([3,5]),
                CNOT([1,6]),
                CNOT([3,4]),
                CNOT([2,5]),
                CNOT([0,6]),
                CNOT([2,4]),
                CNOT([0,5]),
                CNOT([1,4]),
])

def css_decoder(hx: np.ndarray, hz: np.ndarray) -> Circuit:
    gate_list = []
    for i, row in enumerate(hx):
        gate_list.append(H(7+i))
        for j, bit in enumerate(row):
            if bit == 1:
                gate_list.append(CNOT([j, 7+i]))
        gate_list.append(H(7+i))
    
    for i, row in enumerate(hz):
        for j, bit in enumerate(row):
            if bit == 1:
                gate_list.append(CNOT([10+i, j]))
    return Circuit(gate_list)

def pretty_sample(circ: stim.Circuit, num_samples: int = 1000):
    """ prints measurement results """
    samples = circ.compile_sampler().sample(shots=num_samples)
    bitstrings = ["".join(str(int(b)) for b in row) for row in samples]
    counts = Counter(bitstrings)
    bit_vals = sorted(counts.items())
    labels = [bv[0] for bv in bit_vals]
    values = [bv[1] for bv in bit_vals]
    for label, value in zip(labels, values):
        print(f"{label}: {value}")

def css_circuit(enc_stim: Circuit, err_stim: stim.Circuit, hx: np.ndarray, hz=np.ndarray) -> Tuple[stim.TableauSimulator, stim.TableauSimulator]:
    """ generates tableau for a CSS code cycle """
    num_sites = len(hx[0])
    dec_stim = qaravan_to_stim(css_decoder(hx, hz))
    full_stim = enc_stim + err_stim + dec_stim
    full_stim.append('M', [num_sites+i for i in range(6)])
    return full_stim

def cycle(enc_stim: stim.Circuit, full_stim: stim.Circuit, hx: np.ndarray, hz: np.ndarray) -> bool: 
    """ runs a CSS code cycle and returns whether the code is stabilized """
    num_sites = len(hx[0])
    
    full_sim, enc_sim = stim.TableauSimulator(), stim.TableauSimulator()
    full_sim.do_circuit(full_stim)
    enc_sim.do_circuit(enc_stim)
    syndrome = full_sim.current_measurement_record()

    correction = []
    zsyn = np.array(syndrome[:len(hx)], dtype=int)
    xsyn = np.array(syndrome[len(hx):], dtype=int)

    err_loc = np.where((hz.T == zsyn).all(axis=1))[0]
    if len(err_loc) > 0:
        c = ''.join(["Z" if i==err_loc[0] else "I" for i in range(num_sites)])
        correction.append(stim.PauliString(c))

    err_loc = np.where((hx.T == xsyn).all(axis=1))[0]
    if len(err_loc) > 0:
        c = ''.join(["X" if i==err_loc[0] else "I" for i in range(num_sites)])
        correction.append(stim.PauliString(c))

    for corr in correction:
        full_sim.do_pauli_string(corr)

    return truncate_tableau(full_sim.current_inverse_tableau(), num_sites) == enc_sim.current_inverse_tableau() 

def truncate_tableau(tableau: stim.Tableau, n: int) -> stim.Tableau:
    """ truncates a tableau to the first n sites """
    xs = [stim.PauliString(str(tableau.x_output(k))[:n+1]) for k in range(n)]
    zs = [stim.PauliString(str(tableau.z_output(k))[:n+1]) for k in range(n)]
    return stim.Tableau.from_conjugated_generators(xs=xs, zs=zs)