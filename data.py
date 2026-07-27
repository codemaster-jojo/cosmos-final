import numpy as np

def bit_generator(n):
    list_of_bits = []
    for i in range(n):
        for j in range(4096):
            string = ""
            random = round(np.random.default.rng())
            string += str(random)
        string += "\n"
        list_of_bits.append(string)
    with open(list_of_bits.txt, "w", encoding="utf-8") as file:
        file.writelines(list_of_bits)
    return list_of_bits

bit_generator(100000)

    
            
            