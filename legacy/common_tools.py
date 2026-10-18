import numpy as np

# does not work, but worth a try

def infer_precursor_charge(peptide_sequence, matched_ions, precursor_mz=None):

    PROTON_MASS = 1.00794
    WATER_MASS  = 15.9994

    # masses taken from https://proteomicsresource.washington.edu/protocols06/masses.php
    monoisotopic_aa_masses = {
        'A': 71.0779,'R': 156.18568000000002,'N': 114.10264000000001,
        'D': 115.08739999999999,'C': 103.1429,'E': 129.11398,'Q': 128.12922,
        'G': 57.051320000000004,'H': 137.13928,'I': 113.15763999999999,
        'L': 113.15763999999999,'K': 128.17228,'M': 131.19606,'F': 147.17386000000002,
        'P': 97.11518,'S': 87.0773,'T': 101.10388,'W': 186.2099,'Y': 163.17326,
        'V': 99.13105999999999,'O': 237.29815999999997,'U': 150.03789999999998
    }

    # Calculate peptide backbone neutral mass + terminal water
    neutral_peptide_mass = sum(monoisotopic_aa_masses[aa] for aa in peptide_sequence) + WATER_MASS

    # Case 1: Precursor m/z is available (Deterministic solution)
    if precursor_mz is not None and precursor_mz > 0:
        calculated_z = neutral_peptide_mass / (precursor_mz - PROTON_MASS)
        return int(np.round(calculated_z))

    # Case 2: Precursor m/z is missing (Rule-based inference)
    max_fragment_charge = 1
    for ion in matched_ions:
        z_frag = ion.get('charge', 1)
        if z_frag > max_fragment_charge:
            max_fragment_charge = z_frag

    # Count strongly basic residues mapping to CID proton affinity
    basic_residues = sum(peptide_sequence.count(aa) for aa in ['K', 'R', 'H'])
    max_probable_charge = basic_residues + 1 

    inferred_z = max(max_fragment_charge + 1, max_probable_charge)
    
    return int(np.clip(inferred_z, a_min=1, a_max=5))

print(infer_precursor_charge("PYGVLYK", ['y1', 'b2', 'y2', 'b3', 'b4', 'y3', 'y4', 'b5', 'y5', 'b6', 'y6',
       'b7', 'y7']))
