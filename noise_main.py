#PUT IN MAIN
if __name__ == "__main__":
    # symbol_indices: array of ints (0-7), noise_values: array of floats
    # symbol_indices, noise_values = load_your_million_samples(...)
    decoder_model = Decoder().to(device)

decoder_model = Decoder()
decoder_model = decoder_model.to(device)
decoder_model.load_state_dict(torch.load("decoder_irl_test.pth", weights_only = True))
    dataset = NoiseDataset(symbol_indices, noise_values)
    train_set, val_set, test_set = random_split(dataset, [0.8, 0.1, 0.1])

    train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)

    model = NoiseMDN(num_symbols=8, embed_dim=8, hidden=64, K=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    train_hist, val_hist = trainer(model, train_loader, val_loader, optimizer, scheduler,
                                    max_epochs=30, patience=5)

    # reload best checkpoint (not necessarily the last epoch's weights)
    model.load_state_dict(torch.load("noise_mdn_best.pth", weights_only=True))
    print("Training complete. Best model loaded from checkpoint.")