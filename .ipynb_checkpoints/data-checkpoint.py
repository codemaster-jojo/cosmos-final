import numpy as np
from infrastructure import *
import time
import adi

from cosmos import *
from digicomm import *

#NEVER RUN AGAIN!!!!!
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

def add_awgn(symbols, noise_level): # for testing purposes, if necessary
    received = symbols + np.random.randn(len(symbols)) * noise_level

    return received

def send_real_data(constellation):
    target = []

    sdr_tx = adi.Pluto("usb:1.1.5")

    tx = PlutoTransmitter()
    tx.set_sdr(sdr_tx)
    tx.set_channel(9)
    tx.set_power_level(75)

    # l = int(np.log2(len(constellation)))

    
    with open("list_of_bits.txt", "r") as f:
        target = [int(line.strip(), 2) for line in f]

    target_truncated = [target[0], target[1], target[2]] # CHANGE LATER
    
    for t in target_truncated:
        tx.transmit(tx_symbols)
        print("Transmitting data...")
        time.sleep(1)
    

def receive_real_data(constellation, target):
    sdr_rx = adi.Pluto("usb:1.1.5")

    rx = PlutoReceiver()
    rx.set_sdr(sdr_rx)
    rx.set_buffer_size(500e3)
    rx.set_channel(9)
    rx.set_gain_level(80)
    rx.desired_transmit_symbols_real = True
    rx.num_transmit_symbols = 4096/int(np.log2(len(constellation)))

    decoded_arr = []
    
    for i in target:
        rx_symbols = rx.receive()
    
        # REMOVE LATER W/ DECODER MODEL
        decoded = pam_symbols_to_bits(rx_symbols, constellation)
        decoded_arr.append(decoded)

    
    print("Errors: " + sum(np.array(target) == np.array(decoded_arr)))
        

    return rx_symbols, decoded, target