#PUT IN MAIN
if __name__ == "__main__":
    # symbol_indices: array of ints (0-7), noise_values: array of floats
    # symbol_indices, noise_values = load_your_million_samples(...)
    noise_model = NoiseMDN().to(device)

    noise_model = NoiseMDN()
    noise_model = noise_model.to(device)
    noise_model.load_state_dict(torch.load("noise_mdn.pth", weights_only = True))
    
    dataset = NoiseDataset(features, labels)
    train_set, val_set, test_set = random_split(dataset, [0.8, 0.1, 0.1])

    train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False)

    model = NoiseMDN(num_symbols=8, embed_dim=3, hidden=64, K=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    train_hist, val_hist = trainer(model, train_loader, val_loader, optimizer, scheduler)
    #SAVING IS EMBEDDED IN TRAIN FUNCTION

    print("Training complete.")