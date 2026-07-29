import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
def plot_gaussian(weights, means, std_devs):
    # 1. Simulate your real-world ADALM-Pluto SDR noise data
    # We simulate a multimodal distribution (3 peaks) to match your 3 Gaussians
    np.random.seed(42)
    component_1 = np.random.normal(loc=-10.0, scale=2.0, size=5000)   # e.g., Background floor
    component_2 = np.random.normal(loc=0.0, scale=1.5, size=3500)    # e.g., Steady state signal
    component_3 = np.random.normal(loc=8.0, scale=3.5, size=1500)    # e.g., High-power bursts
    raw_sdr_data = np.concatenate([component_1, component_2, component_3])
    
    # 2. Define the dummy outputs from your MDN model for a specific input location
    # Replace these placeholder arrays with your actual model's tensor predictions!
    # Note: Mixing weights (pi) must sum to 1.0.
    weights = weights.detach().cpu().numpy()
    std_devs = std_devs.detach().cpu().numpy()
    means = means.detach().cpu().numpy()
    
    # 3. Generate a smooth x-axis grid for plotting the continuous mathematical curve
    x_axis = np.linspace(raw_sdr_data.min() - 5, raw_sdr_data.max() + 5, 1000)
    
    # 4. Calculate the individual curves and the total blended MDN curve
    individual_gaussians = []
    total_mdn_pdf = np.zeros_like(x_axis)
    
    for i in range(3):
        # Calculate the PDF for this specific Gaussian component, scaled by its mixing weight
        component_pdf = weights[i] * norm.pdf(x_axis, loc=means[i], scale=std_devs[i])
        individual_gaussians.append(component_pdf)
        # Add it to the total mixture curve
        total_mdn_pdf += component_pdf
    
    # 5. Create the visual plot
    plt.figure(figsize=(10, 6), dpi=100)
    
    # Plot the real raw data as a density histogram (density=True ensures it matches the math curve scale)
    plt.hist(raw_sdr_data, bins=80, density=True, alpha=0.4, color='gray', label='Raw SDR Noise Data')
    
    # Plot each individual Gaussian component as a dashed line to see how the work is split
    colors = ['blue', 'orange', 'green']
    for i in range(3):
        plt.plot(x_axis, individual_gaussians[i], linestyle='--', color=colors[i], 
                 label=r'Gaussian %d ($\pi$=%.2f, $\mu$=%.1f, $\sigma$=%.1f)' % (i+1, pi_weights[i], means[i], std_devs[i]))
    
    # Plot the final combined MDN distribution model
    plt.plot(x_axis, total_mdn_pdf, color='red', linewidth=2.5, label='Total Predicted MDN Distribution')
    
    # Add styling and labels for your engineering report
    plt.title('MDN 3-Gaussian Fit vs. ADALM-Pluto SDR Noise Floor (Validation NLL: 0.5)', fontsize=14, fontweight='bold')
    plt.xlabel('Signal Amplitude / Power Value', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', fontsize=10)
    
    # Render the graph
    plt.tight_layout()
    plt.show()