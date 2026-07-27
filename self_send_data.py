import threading
import time
import adi
import numpy as np

from infrastructure import *
from cosmos import *
from digicomm import *

constellation = np.array([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0])
constellation /= np.sqrt(np.mean(constellation ** 2))
bits_per_symbol = int(np.log2(len(constellation)))

with open("list_of_bits.txt") as f:
    packets = [line.strip() for line in f][:50]

NUM_SYMBOLS = len(packets[0]) // bits_per_symbol

ack = threading.Event()
tx_live = threading.Event()
running = True
BER_ACK_THRESHOLD = 0.3


def ber(tx_bits, rx_bits):
    n = min(len(tx_bits), len(rx_bits))
    if n == 0:
        return 1.0, 0, 0
    err = sum(a != b for a, b in zip(tx_bits[:n], rx_bits[:n]))
    return err / n, err, n


def tx_loop():
    global running

    tx = PlutoTransmitter()
    tx.set_sdr(adi.Pluto("usb:1.1.5"))
    tx.set_channel(9)
    tx.set_power_level(95)

    for i, bits in enumerate(packets):
        symbols = bits_to_pam_symbols(bits, constellation)
        if len(symbols) < NUM_SYMBOLS:
            symbols = np.pad(symbols, (0, NUM_SYMBOLS - len(symbols)))
        else:
            symbols = symbols[:NUM_SYMBOLS]

        while running:
            print(f"\n========== PACKET {i} ==========")
            #print("TX:", np.round(symbols[:30], 3))

            try:
                try:
                    tx.sdr.tx_destroy_buffer()
                except Exception:
                    pass
                tx.transmit(symbols)
                tx_live.set()
            except Exception as e:
                #print("TX error:", e)
                tx_live.clear()
                time.sleep(0.5)
                continue

            #print("Waiting for ACK...")
            if ack.wait(timeout=8):
                ack.clear()
                tx_live.clear()
                try:
                    tx.sdr.tx_destroy_buffer()
                except Exception:
                    pass
                #print(f"Packet {i} confirmed")
                time.sleep(0.2)
                break

            #print("No ACK, retransmitting")
            tx_live.clear()
            try:
                tx.sdr.tx_destroy_buffer()
            except Exception:
                pass
            time.sleep(0.2)

    running = False
    tx_live.clear()
    try:
        tx.sdr.tx_destroy_buffer()
    except Exception:
        pass
    print("Finished sending packets")


def rx_loop():
    global running

    rx = PlutoReceiver()
    rx.set_sdr(adi.Pluto("usb:0.1.5"))
    rx.set_buffer_size(500000)
    rx.set_channel(9)
    rx.set_gain_level(80)
    rx.desired_transmit_symbols_real = True
    rx.num_transmit_symbols = NUM_SYMBOLS

    packet_idx = 0

    while running and packet_idx < len(packets):
        # Only capture while TX is actively looping the current packet
        if not tx_live.wait(timeout=0.1):
            continue

        try:
            rx_symbols = np.real(rx.receive())
            detected = pam_detect(rx_symbols, constellation)
            rx_bits = "".join(str(b) for b in pam_symbols_to_bits(detected, constellation))
            tx_bits = packets[packet_idx]
            rate, err, n = ber(tx_bits, rx_bits)

            #print("RX:", np.round(rx_symbols[:30], 3))
            #print("TX bits:", tx_bits[:100])
            #print("RX bits:", rx_bits[:100])
            print(f"BER={err}/{n}={rate:.3f}")

            if rate <= BER_ACK_THRESHOLD:
                ack.set()
                packet_idx += 1
                # Wait until TX stops this packet before next capture
                while tx_live.is_set() and running:
                    time.sleep(0.05)
            else:
                #print("BER too high, not ACKing (will retry)")
                time.sleep(0.1)

        except Exception as e:
            #print("RX error:", e)
            time.sleep(0.1)


#basically have 2 loops running at once so i can send and receive at the same time
rx_thread = threading.Thread(target=rx_loop)
tx_thread = threading.Thread(target=tx_loop)

rx_thread.start()
time.sleep(1)
tx_thread.start()

tx_thread.join()
running = False
tx_live.set()
rx_thread.join(timeout=2)

print("Done")
