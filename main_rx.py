import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import time 
import sys

from cosmos import *
from digicomm import *

import adi

# Directory for saving plots
dir_plots = 'plots/'

# ---------------------------------------------------------------
# Setup.
# ---------------------------------------------------------------
sdr_rx = adi.Pluto("usb:1.1.5")

rx = PlutoReceiver()
rx.set_sdr(sdr_rx)
rx.set_buffer_size(500e3)
rx.set_channel(9)
rx.set_gain_level(80)
rx.desired_transmit_symbols_real = True # set to false for imaginary symbols
rx.num_transmit_symbols = 200

# print(rx.sdr)
# ---------------------------------------------------------------
# Receive.
# ---------------------------------------------------------------
rx_symbols = rx.receive()

if True:
    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(rx_symbols),np.imag(rx_symbols), color='red', label='Received PAM Symbols')
    plt.title('Data Symbols After Equalization')
    plt.xlabel('Real Component')
    plt.ylabel('Imaginary Component')
    plt.grid(True)
    plt.legend()
    plt.show()
