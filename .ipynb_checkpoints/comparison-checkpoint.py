import time
import adi
import numpy as np

from infrastructure import *
from cosmos import *
from digicomm import *
from data import *

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

def calculate_ber(constellation, n=10):
    with open("list_of_bits.txt", "r") as f:
        packets = [line.strip() for line in f if line.strip()]

    selected = np.random.choice(packets, size=min(n, len(packets)), replace=False)
    constellation = np.asarray(constellation, dtype=float)
    bers = []

    for bits in selected:
        symbols = bits_to_pam_symbols(bits, constellation)
        rx.num_transmit_symbols = len(symbols)

        try:
            tx.sdr.tx_destroy_buffer()
        except Exception:
            pass

        tx.transmit(symbols)
        rx_symbols = np.real(rx.receive())

        detected = pam_detect(rx_symbols, constellation)
        rx_bits = pam_symbols_to_bits(detected, constellation)
        bits_per_symbol = int(np.log2(len(constellation)))
        tx_bits = np.array([int(b) for b in bits[: len(detected) * bits_per_symbol]])

        n_cmp = min(len(tx_bits), len(rx_bits))
        if n_cmp == 0:
            bers.append(1.0)
        else:
            bers.append(calculate_BER(rx_bits[:n_cmp], tx_bits[:n_cmp]))

    return float(np.mean(bers))

standard = calculate_ber([
                -1.5275,
                -1.0911,
                -0.6547,
                -0.2182,
                 0.2182,
                 0.6547,
                 1.9011,
                 1.5275
            ])
print("Standard constellation BER:", standard)

model = calculate_ber([1.5275, -1.0911, -0.6547, -0.2182,  0.2182,  0.6547,  1.0911,  1.5275])

print("Model constellation BER:   ", standard)
