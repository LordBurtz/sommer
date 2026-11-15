"""Task-wide constants: token vocabulary, b/y vector layout, charges, masses."""

MAX_PEPTIDE_LEN = 30
# A peptide of length L produces L-1 b ions and L-1 y ions.
MAX_FRAGMENTS = MAX_PEPTIDE_LEN - 1            # 29 fragment numbers
ION_TYPES = ("b", "y")
N_ION_TYPES = len(ION_TYPES)
VECTOR_DIM = MAX_FRAGMENTS * N_ION_TYPES       # 58 slots per spectrum

# 1, 5, 6 are dropped (too few spectra)
CHARGES = (2, 3, 4)

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
MOD_TOKENS = ["C[UNIMOD:4]", "M[UNIMOD:35]"]

PAD_IDX = 0
TOKENS = ["<pad>"] + AMINO_ACIDS + MOD_TOKENS
TOKEN_TO_IDX = {tok: i for i, tok in enumerate(TOKENS)}
VOCAB_SIZE = len(TOKENS)
MOD_TOKEN_IDX = [TOKEN_TO_IDX[t] for t in MOD_TOKENS]

AA_BASIC = set("KRH")
AA_ACIDIC = set("DE")

# monoisotopic residue masses, as taken from
# https://proteomicsresource.washington.edu/resources/data-analysis/masses/
AA_MONO_MASS = {
    'A': 71.037114, 'R': 156.101111, 'N': 114.042927, 'D': 115.026943,
    'C': 103.009185, 'E': 129.042593, 'Q': 128.058578, 'G': 57.021464,
    'H': 137.058912, 'I': 113.084064, 'L': 113.084064, 'K': 128.094963,
    'M': 131.040485, 'F': 147.068414, 'P': 97.052764, 'S': 87.032028,
    'T': 101.047679, 'W': 186.079313, 'Y': 163.063329, 'V': 99.068414,
    'O': 237.147727, 'U': 150.953633,
}
MOD_MASS = {"C[UNIMOD:4]": 57.02146, "M[UNIMOD:35]": 15.99491}
WATER_MASS = 18.01056
