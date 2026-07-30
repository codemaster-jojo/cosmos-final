import time
import adi
import numpy as np

import infrastructure
from cosmos import *
from digicomm import *
from data import *

tx = PlutoTransmitter()
sdr_tx = adi.Pluto("usb:0.1.5")
tx.set_sdr(sdr_tx)
tx.set_channel(9)
tx.set_power_level(90) # 70

sdr_rx = adi.Pluto("usb:1.1.5")

rx = PlutoReceiver()
rx.set_sdr(sdr_rx)
rx.set_buffer_size(1e6)
rx.set_channel(9)
rx.set_gain_level(70) # 60
rx.desired_transmit_symbols_real = False

def calculate_ber(constellation, n=50):
    with open("list_of_bits.txt", "r") as f:
        packets = [line.strip() for line in f if line.strip()]

    selected = np.random.choice(packets, size=min(n, len(packets)), replace=False)
    constellation = np.asarray(constellation, dtype=complex)
    bers = []

    for bits in selected:
        symbols = infrastructure.bits_to_qam_symbols(bits, constellation)
        rx.num_transmit_symbols = len(symbols)

        try:
            tx.sdr.tx_destroy_buffer()
        except Exception:
            pass

        tx.transmit(symbols)
        rx_symbols = np.real(rx.receive())

        detected = infrastructure.qam_detect(rx_symbols, constellation)
        rx_bits = infrastructure.qam_symbols_to_bits(detected, constellation)
        bits_per_symbol = int(np.log2(len(constellation)))
        tx_bits = np.array([int(b) for b in bits[: len(detected) * bits_per_symbol]])

        n_cmp = min(len(tx_bits), len(rx_bits))
        if n_cmp == 0:
            bers.append(1.0)
        else:
            bers.append(infrastructure.calculate_BER(rx_bits[:n_cmp], tx_bits[:n_cmp]))

    return float(np.mean(bers))

standard = calculate_ber([
        -7 - 7j, -7 - 5j, -7 - 3j, -7 - 1j, -7 + 1j, -7 + 3j, -7 + 5j, -7 + 7j,
        -5 - 7j, -5 - 5j, -5 - 3j, -5 - 1j, -5 + 1j, -5 + 3j, -5 + 5j, -5 + 7j,
        -3 - 7j, -3 - 5j, -3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j, -3 + 5j, -3 + 7j,
        -1 - 7j, -1 - 5j, -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j, -1 + 5j, -1 + 7j,
         1 - 7j,  1 - 5j,  1 - 3j,  1 - 1j,  1 + 1j,  1 + 3j,  1 + 5j,  1 + 7j,
         3 - 7j,  3 - 5j,  3 - 3j,  3 - 1j,  3 + 1j,  3 + 3j,  3 + 5j,  3 + 7j,
         5 - 7j,  5 - 5j,  5 - 3j,  5 - 1j,  5 + 1j,  5 + 3j,  5 + 5j,  5 + 7j,
         7 - 7j,  7 - 5j,  7 - 3j,  7 - 1j,  7 + 1j,  7 + 3j,  7 + 5j,  7 + 7j,
    ])

model = calculate_ber([-1.1812-1.0576j, -1.0556-0.6808j, -1.3041-0.3665j, -1.0127-0.1292j,
        -1.3094+0.1083j, -1.0225+0.3750j, -1.2391+0.6581j, -1.1016+1.0646j,
        -0.7667-1.1115j, -0.6851-0.7189j, -0.8431-0.4028j, -0.6589-0.1356j,
        -0.8282+0.1208j, -0.6526+0.4044j, -0.8085+0.6962j, -0.7155+1.0873j,
        -0.4286-1.0708j, -0.3771-0.6931j, -0.5113-0.4055j, -0.3558-0.1429j,
        -0.5047+0.1248j, -0.3580+0.4034j, -0.4594+0.7504j, -0.3950+1.2189j,
        -0.1378-1.2184j, -0.1089-0.7620j, -0.2121-0.4178j, -0.0769-0.1373j,
        -0.2195+0.1198j, -0.0657+0.3756j, -0.1768+0.6525j, -0.1394+0.9969j,
         0.1341-1.0145j,  0.1662-0.6652j,  0.0570-0.3900j,  0.2066-0.1439j,
         0.0621+0.1146j,  0.2036+0.4037j,  0.1030+0.7564j,  0.1096+1.2828j,
         0.3902-1.2668j,  0.4471-0.7655j,  0.3528-0.4143j,  0.4990-0.1402j,
         0.3417+0.1249j,  0.5099+0.3862j,  0.3866+0.6599j,  0.3701+1.0188j,
         0.6840-1.0521j,  0.7862-0.6857j,  0.6565-0.4012j,  0.8405-0.1395j,
         0.6451+0.1211j,  0.8408+0.4104j,  0.7021+0.7408j,  0.6862+1.1972j,
         1.0757-1.1130j,  1.2157-0.7142j,  1.0571-0.3942j,  1.3056-0.1015j,
         0.9969+0.1304j,  1.3196+0.3492j,  1.1046+0.6666j,  1.0750+1.0580j])

print("Standard constellation BER:", standard)
print("Model constellation BER:   ", model)
