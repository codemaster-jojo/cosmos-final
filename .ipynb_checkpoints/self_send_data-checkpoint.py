import threading
import numpy as np
import time
import adi

from infrastructure import *
from cosmos import *
from digicomm import *


received_flag = threading.Event()
running = True

#constellation = [-1.5, -1.1, -0.65, -0.22, 0.22, 0.65, 1.1, 1.5]
constellation = [
    -7/np.sqrt(21),
    -5/np.sqrt(21),
    -3/np.sqrt(21),
    -1/np.sqrt(21),
     1/np.sqrt(21),
     3/np.sqrt(21),
     5/np.sqrt(21),
     7/np.sqrt(21)
]
NUM_SYMBOLS = 4096 // int(np.log2(len(constellation)))


def tx_loop():
    global running

    with open("list_of_bits.txt", "r") as f:
        target = [line.strip() for line in f][:50]

    tx = PlutoTransmitter()
    tx.set_sdr(adi.Pluto("usb:1.1.5"))
    tx.set_channel(9)
    tx.set_power_level(95)

    tx.desired_transmit_symbols_real = True
    tx.num_transmit_symbols = NUM_SYMBOLS

    for i, bits in enumerate(target):

        while True:
            print("Transmitting packet", i)

            symbols = bits_to_pam_symbols(bits, constellation)

            # pad/truncate to expected frame size
            if len(symbols) < NUM_SYMBOLS:
                symbols = np.pad(
                    symbols,
                    (0, NUM_SYMBOLS - len(symbols))
                )
            else:
                symbols = symbols[:NUM_SYMBOLS]


            try:
                tx.transmit(symbols)

                try:
                    tx.sdr.tx_destroy_buffer()
                except:
                    pass

            except Exception as e:
                print("TX error:", e)
                time.sleep(0.5)
                continue


            print("Waiting for ACK...")

            if received_flag.wait(timeout=5):
                received_flag.clear()

                print("Packet", i, "confirmed")
                print(symbols)

                time.sleep(0.1)
                break

            else:
                print("No ACK, retransmitting packet", i)


    print("Finished sending 50 packets")
    running = False



def rx_loop():
    global running

    rx = PlutoReceiver()
    rx.set_sdr(adi.Pluto("usb:0.1.5"))
    rx.set_buffer_size(500000)
    rx.set_channel(9)
    rx.set_gain_level(60)

    rx.desired_transmit_symbols_real = True
    rx.num_transmit_symbols = NUM_SYMBOLS


    while running:

        print("Waiting for packet...")

        try:
            rx_symbols = rx.receive()

            print("Received symbols:")
            print(np.real(rx_symbols))

            # fake ACK
            received_flag.set()

        except Exception as e:
            print("RX error:", e)
            time.sleep(0.1)



rx_thread = threading.Thread(target=rx_loop)
tx_thread = threading.Thread(target=tx_loop)


# Start RX first
rx_thread.start()
time.sleep(1)

tx_thread.start()


tx_thread.join()

running = False

rx_thread.join(timeout=2)

print("Done")