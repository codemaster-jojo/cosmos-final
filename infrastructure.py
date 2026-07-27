import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#utility functions

#bits_and_pam
def get_pam_constellation(M, Es=1):
    """
    Generate an M-PAM constellation.
    """
    # Add your constellation code here.
    length = int(np.log2(M))
    if 2 ** length != M:
        print("Give power of 2")
        return
        
    const = np.full(M, "", dtype=object)
    const[0] = "0" * length
    const[1] = "0" * (length - 1) + "1"

    
    index = 2
    while (2 * index <= M):
        old = const[:index][::-1]
        
        index_flip = length - int(np.log2(index)) - 1

        const[index:2*index] = [c[:index_flip] + "1" + c[index_flip+1:] for c in old]
        index *= 2

    # sum

    const_values = np.zeros(M)

    s = (M**2 - 1) / 3 # average
    factor = np.sqrt(Es / s)
    
    curr = -(M-1)
    
    for c in const:
        const_values[int(c, 2)] = curr * factor
        curr += 2
    
    
    return const_values

def bits_to_pam_symbols(bits, M):
    """
    Convert a bit sequence to a sequence of M-PAM symbols.
    """
    # Add your conversion code here.
    length = int(np.log2(M))
    symbols = np.zeros(int(len(bits) // length))

    constellation = get_pam_constellation(M)

    
    for i in range(len(symbols)):
        symbol = bits[i*length:i*length+length]
        result = int("".join(map(str, symbol)), 2)
        symbols[i] = constellation[result]

    return symbols


def pam_symbols_to_bits(symbols, M):
    """
    Convert a sequence of M-PAM symbols back to a bit sequence.
    """
    # Add your conversion code here.
    bits = []

    constellation = get_pam_constellation(M)

    reverse_const = dict(zip(constellation, np.arange(0, M+1)))
    
    for symbol in symbols:
        binary = reverse_const[symbol]

        binary_array_bits = [(binary >> i) & 1 for i in reversed(range(int(np.log2(M))))]

        bits += binary_array_bits

    bits = np.array(bits)
        

    
    
    return bits

#files_and_bytes
def file_to_bytes(file_path):
    """
    Read a file's raw bytes.
    """
    with open(file_path, 'rb') as f:
        return f.read()

def bytes_to_bmp_image(byte_data):
    byte_data_io = io.BytesIO(byte_data)
    img = Image.open(byte_data_io)
    return np.array(img)


#detection
def get_decision_boundaries(constellation):
    """
    Midpoints between adjacent constellation points, dividing maximum
    likelihood decision regions.
    """
    # Zeros as placeholder.
    constellation.sort()
    boundaries = np.ones(constellation.size - 1)

    for i in range(len(constellation)-1):
        boundaries[i] = (constellation[i] + constellation[i+1])/2
    
    return boundaries

def pam_detect(symbols, M):
    """
    Detect noisy M-PAM symbols.
    """

    # Find the M-PAM constellation (i.e., all of the possible symbols).
    constellation = get_pam_constellation(M)
    constellation.sort()
    boundaries = get_decision_boundaries(constellation)
    
    # Choose the nearest constellation symbol for each noisy symbol.
    detected_symbols = np.zeros_like(symbols)
    for i,symbol in enumerate(symbols):
        less_than = symbol < boundaries
        indices = np.flatnonzero(less_than)
        first_true = indices[0] if indices.size else M-1
        detected_symbols[i] = constellation[first_true]
    return detected_symbols

#bits_and_bytes
def bytes_to_bits(byte_data):
    """
    Convert raw bytes into a bit sequence.
    """
    # Add your conversion code here.
    byte_array = np.frombuffer(byte_data, dtype=np.uint8)
    return np.unpackbits(byte_array)

def bits_to_bytes(bits):
    """
    Convert a bit sequence back into raw bytes.
    """
    # Add your conversion code here.
    return np.packbits(bits.astype(np.uint8)).tobytes()
