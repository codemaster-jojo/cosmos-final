import time
import adi
import numpy as np

from infrastructure import *
from cosmos import *
from digicomm import *
from data import *

#google says this will make it print not looping
np.set_printoptions(linewidth=np.inf)


num_symbols = 700

tx = PlutoTransmitter()
tx.set_sdr(adi.Pluto("usb:1.1.5"))
tx.set_channel(9)
tx.set_power_level(90)

rx = PlutoReceiver()
rx.set_sdr(adi.Pluto("usb:0.1.5"))
rx.set_buffer_size(500000)
rx.set_channel(9)
rx.set_gain_level(70)
rx.desired_transmit_symbols_real = True
rx.num_transmit_symbols = num_symbols

for i in range(1):
    # symbols = generate_data(num_symbols)
    symbols = np.array([-3, -2, -1, 0, 1, 2, 3] * 100, dtype=float)

    tx.transmit(symbols)

    received = rx.receive()

    # print("Transmitted:", symbols)
    # print("Received:   ", received)
    # print("Noise:      ", received - symbols)

    if True:
        plt.figure(figsize=(6, 6))
        plt.scatter(np.real(received),np.imag(received), color='red', label='Received PAM Symbols')
        plt.title('Data Symbols After Equalization')
        plt.xlabel('Real Component')
        plt.ylabel('Imaginary Component')
        plt.grid(True)
        plt.legend()
        plt.show()



