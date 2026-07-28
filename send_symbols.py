import time
import adi
import numpy as np

from infrastructure import *
from cosmos import *
from digicomm import *
from data import *

#google says this will make it print not looping
np.set_printoptions(linewidth=np.inf)


num_repeat = 1000
num_symbols = 8 * num_repeat

tx = PlutoTransmitter()
sdr_tx = adi.Pluto("usb:1.1.5")
tx.set_sdr(sdr_tx)
tx.set_channel(9)
tx.set_power_level(90) # 70

sdr_rx = adi.Pluto("usb:0.1.5")

rx = PlutoReceiver()
rx.set_sdr(sdr_rx)
rx.set_buffer_size(1e6)
rx.set_channel(9)
rx.set_gain_level(70) # 60
# rx.set_receive_gain(rx_gain_dB)
rx.desired_transmit_symbols_real = True
rx.num_transmit_symbols = num_symbols

print(tx.sdr)

print(rx.sdr)

received = []
for i in range(10):
    tx.sdr.tx_destroy_buffer()
    
    # symbols = generate_data(num_symbols)
    rng = np.random.default_rng()
    
    symbols = np.array([
                -1.5275252316519465,
                -1.0910894511799618,
                -0.6546536707079771,
                -0.21821789023599236,
                 0.21821789023599236,
                 0.6546536707079771,
                 1.0910894511799618,
                 1.5275252316519465
            ] * num_repeat, dtype=complex)
    rng.shuffle(symbols)
    

    tx.transmit(symbols)
    time.sleep(1)
    
    # received = rx.receive()

    # print("Transmitted:", symbols)
    # print("Received:   ", received)
    # print("Noise:      ", received - symbols)

    received = np.concatenate((received, rx.receive()))

if True:
    plt.figure(figsize=(6, 6))
    # plt.scatter(np.real(received),np.imag(received), color='red', label='Received PAM Symbols')
    plt.hist(np.real(received), bins=500, label='Received PAM Symbols')
    plt.title('Data Symbols After Equalization')
    plt.xlabel('Symbol Received')
    plt.ylabel('Number of Times Received')
    plt.grid(True)
    plt.legend()
    plt.show()