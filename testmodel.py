import torch

from model import Constellation, Decoder


def main():
    constellation = Constellation()
    decoder = Decoder()

    # Test all 8 symbols
    symbol_indices = torch.arange(8)

    # Map indices to PAM values
    tx_symbols = constellation(symbol_indices)

    print("Constellation:")
    for i, s in enumerate(tx_symbols):
        print(f"Symbol {i}: {s.item():.4f}")


    # Decode without noise
    decoded = decoder.decode_hard(tx_symbols)

    print("\nNo Noise Test")
    print("----------------")
    print("True:    ", symbol_indices.tolist())
    print("Decoded: ", decoded.tolist())

    if torch.equal(symbol_indices, decoded):
        print("PASS: Perfect decoding without noise!")
    else:
        print("FAIL")

    # Decode with Gaussian noise
    noise_std = 0.15
    noisy_symbols = tx_symbols + noise_std * torch.randn_like(tx_symbols)

    decoded_noisy = decoder.decode_hard(noisy_symbols)

    print("\nWith Noise")
    print("----------------")

    for i in range(8):
        print(
            f"True={symbol_indices[i].item()} "
            f"Tx={tx_symbols[i].item():6.3f} "
            f"Rx={noisy_symbols[i].item():6.3f} "
            f"Decoded={decoded_noisy[i].item()}"
        )

    num_correct = (decoded_noisy == symbol_indices).sum().item()

    print(f"\nCorrect: {num_correct}/8")


if __name__ == "__main__":
    main()