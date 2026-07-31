# send image
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import adi

import infrastructure
from cosmos import *


def send_receive_image(constellation, img_path):
    original_image = np.array(Image.open(img_path))

    bytes_file = infrastructure.file_to_bytes(img_path)
    bits_file = infrastructure.bytes_to_bits(bytes_file)
    symbols_send = infrastructure.bits_to_qam_symbols(bits_file, constellation)

    symbols_per_packet = 8000

    symbol_chunks = [symbols_send[i:i + symbols_per_packet] for i in range(0, len(symbols_send), symbols_per_packet)]

    tx = PlutoTransmitter()
    sdr_tx = adi.Pluto("usb:0.1.5")
    tx.set_sdr(sdr_tx)
    tx.set_channel(9)
    tx.set_power_level(90)
    
    sdr_rx = adi.Pluto("usb:1.1.5")
    
    rx = PlutoReceiver()
    rx.set_sdr(sdr_rx)
    rx.set_buffer_size(1e6)
    rx.set_channel(9)
    rx.set_gain_level(70)
    rx.desired_transmit_symbols_real = False # this changed to false so it receives everything
    rx.num_transmit_symbols = len(symbols_send)


    received_symbols = []

    for chunk in symbol_chunks:
        try:
            tx.sdr.tx_destroy_buffer()
        except Exception:
            pass

        
        rx.num_transmit_symbols = len(chunk)
    
        tx.transmit(chunk)
        received = rx.receive()
    
        received_symbols.append(received)

    received_symbols = np.concatenate(received_symbols)

    detected = infrastructure.qam_detect(received_symbols, constellation)
    received_bits = infrastructure.qam_symbols_to_bits(detected, constellation)
    received_bytes = infrastructure.bits_to_bytes(received_bits)
    recovered_image = infrastructure.bytes_to_bmp_image(received_bytes)

    # plot
    plt.imshow(recovered_image)
    plt.title("Recovered Image")
    plt.show()

send_receive_image(np.array([-1.0879-0.9933j, -1.2819-0.5974j, -0.9351-0.4130j, -1.2642-0.2118j,
        -1.2346+0.1023j, -1.0201+0.3847j, -1.2573+0.6607j, -1.1210+1.0568j,
        -0.6935-1.1228j, -0.7977-0.7166j, -0.6067-0.4127j, -0.8009-0.1383j,
        -0.8193+0.1373j, -0.6569+0.4171j, -0.8196+0.7120j, -0.7285+1.1182j,
        -0.3559-1.0901j, -0.4668-0.7271j, -0.3158-0.4355j, -0.4714-0.1417j,
        -0.5025+0.1356j, -0.3606+0.4072j, -0.4744+0.7423j, -0.4039+1.1636j,
        -0.0770-1.3065j, -0.1574-0.7409j, -0.0439-0.4095j, -0.1913-0.1657j,
        -0.2102+0.1168j, -0.0888+0.4047j, -0.1746+0.6946j, -0.1188+1.0638j,
         0.0965-0.9499j,  0.1539-0.6133j,  0.2247-0.3084j,  0.0654-0.0899j,
         0.0784+0.1842j,  0.1764+0.4622j,  0.1160+0.7894j,  0.1607+1.3035j,
         0.3237-1.3073j,  0.3945-0.8381j,  0.4458-0.4967j,  0.4931-0.1757j,
         0.3184+0.0521j,  0.4312+0.3270j,  0.4143+0.6347j,  0.3910+0.9988j,
         0.6511-1.1009j,  0.7303-0.7068j,  0.7462-0.3883j,  0.8467-0.1121j,
         0.6189+0.1042j,  0.7461+0.4314j,  0.7160+0.7782j,  0.6901+1.2361j,
         1.0461-1.1095j,  1.1331-0.6942j,  1.1462-0.3589j,  1.2880-0.0428j,
         0.9507+0.1815j,  1.3568+0.3124j,  1.1071+0.6062j,  1.0630+1.0163j]
), "bmp_24.bmp")