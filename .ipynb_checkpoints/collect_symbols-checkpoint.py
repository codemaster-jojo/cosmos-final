import time
import adi
import numpy as np

from infrastructure import *
from cosmos import *
from digicomm import *
from data import *

np.set_printoptions(linewidth=np.inf)

num_repeat = 1000
num_symbols = 8 * num_repeat

tx = PlutoTransmitter()
sdr_tx = adi.Pluto("usb:1.1.5")
tx.set_sdr(sdr_tx)
tx.set_channel(9)
tx.set_power_level(90)

sdr_rx = adi.Pluto("usb:0.1.5")

rx = PlutoReceiver()
rx.set_sdr(sdr_rx)
rx.set_buffer_size(1e6)
rx.set_channel(9)
rx.set_gain_level(70)
rx.desired_transmit_symbols_real = True
rx.num_transmit_symbols = num_symbols


def collect(n):
    symbols_in = generate_data(n)
    tx.transmit(symbols_in)
    symbols_out = rx.receive()

    return symbols_in, symbols_out


symbol_in_list = []
symbol_out_list = []

for i in range(150):
    print(f"Collecting batch {i + 1}...")
    tx.sdr.tx_destroy_buffer()
    sin, sout = collect(8000)
    symbol_in_list.append(sin)
    symbol_out_list.append(sout)

symbol_in = np.concatenate(symbol_in_list)
symbol_out = np.concatenate(symbol_out_list)

with open("signals_through_wire.txt", "w") as f:
    for s_in, s_out in zip(symbol_in, symbol_out):
        f.write(f"{s_in} {s_out}\n")

print(f"Saved {len(symbol_in)} symbol pairs to signals_through_wire.txt")