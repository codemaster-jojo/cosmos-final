import numpy as np

def bit_generator(n):
    list_of_bits = []
    for i in range(n):
        string = ""
        for j in range(4096):
            random = round(np.random.rand())
            string += str(random)
        string += "\n"
        list_of_bits.append(string)
    with open("list_of_bits.txt", "w", encoding="utf-8") as file:
        file.writelines(list_of_bits)
    return list_of_bits

def add_awgn(symbols, noise_level): # for testing purposes, if necessary
    received = symbols + np.random.randn(len(symbols)) * noise_level

    return received

def collect_real_data():
    target = []
    received = []