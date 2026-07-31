import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#utility functions
# NEEDS CLEANING -- UNNECESSARY DUPING

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

def bits_to_pam_symbols(bits, constellation):
    length = int(np.log2(len(constellation)))
    num_symbols = len(bits) // length

    symbols = np.zeros(num_symbols)

    for i in range(num_symbols):
        chunk = bits[i*length:(i+1)*length]
        index = int(chunk, 2)
        symbols[i] = constellation[index]

    return symbols

# same method, just renamed
def bits_to_qam_symbols(bits, constellation):
    length = int(np.log2(len(constellation)))
    num_symbols = len(bits) // length

    symbols = np.zeros(num_symbols, dtype=constellation.dtype)

    for i in range(num_symbols):
        chunk = bits[i*length:(i+1)*length]

        if isinstance(chunk, str):
            index = int(chunk, 2)
        else:
            index = 0
            for bit in chunk:
                index = (index << 1) | int(bit)

        symbols[i] = constellation[index]

    return symbols


def pam_symbols_to_bits(symbols, constellation):
    bits = []

    length = int(np.log2(len(constellation)))

    for symbol in symbols:
        index = np.argmin(np.abs(np.array(constellation) - symbol))

        binary_bits = [
            int(x) for x in format(index, f'0{length}b')
        ]

        bits.extend(binary_bits)

    return np.array(bits)

# same function, just renamed
def qam_symbols_to_bits(symbols, constellation):
    bits = []

    length = int(np.log2(len(constellation)))

    for symbol in symbols:
        index = np.argmin(np.abs(np.array(constellation) - symbol))

        binary_bits = [
            int(x) for x in format(index, f'0{length}b')
        ]

        bits.extend(binary_bits)

    return np.array(bits)
    
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

def pam_detect(symbols, c):
    """
    Detect noisy M-PAM symbols.
    """

    # Find the M-PAM constellation (i.e., all of the possible symbols).
    constellation = np.sort(np.asarray(c, dtype=float))
    boundaries = get_decision_boundaries(constellation.copy())
    
    # Choose the nearest constellation symbol for each noisy symbol.
    detected_symbols = np.zeros_like(symbols)
    for i,symbol in enumerate(symbols):
        less_than = symbol < boundaries
        indices = np.flatnonzero(less_than)
        first_true = indices[0] if indices.size else len(c)-1
        detected_symbols[i] = constellation[first_true]
    return detected_symbols

# nearest neighbor instead of decision boundaries
def qam_detect(symbols, constellation):
    # get nearest neighbor
    detected_symbols = np.zeros_like(symbols, dtype=constellation.dtype)

    for i, symbol in enumerate(symbols):
        index = np.argmin(np.abs(constellation - symbol))
        detected_symbols[i] = constellation[index]

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

def bits_to_symbol_indices(bits, M):
    length = int(np.log2(M))
    num_symbols = len(bits) // length
    symbols = np.zeros(num_symbols, dtype=np.int64)

    for i in range(num_symbols):
        segment = bits[i*length:(i+1)*length]
        # Convert array of bits into a string
        string = ""
        for bit in segment:
            string += str(bit)
        symbols[i] = int(string, 2)
    return symbols

def calculate_BER(received, expected):
    ber = 0
    for i in range(len(received)):
        if received[i] != expected[i]:
            ber += 1
    return ber / len(received)

def pam_from_indices(constellation, indices):
    pam_symbols = []
    for i in range(len(indices)):
        pam_symbols.append(constellation[indices[i]])
    return pam_symbols

# works for qam too
def qam_from_indices(constellation, indices):
    return constellation[indices]

def apply_gain(symbols): # for pam only
    return symbols / (1 + 0.5*symbols**2)