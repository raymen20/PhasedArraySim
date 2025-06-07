import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, FloatSlider

def simulate_pressure_field(num_elements=8, steering_angle_deg=30, freq=40000, spacing_factor=0.5):
    # Constants
    c = 343
    wavelength = c / freq
    k = 2 * np.pi / wavelength
    omega = 2 * np.pi * freq
    t = 0
    d = spacing_factor * wavelength

    # Grid
    x_range = np.linspace(-0.2, 0.2, 400)
    y_range = np.linspace(0.01, 0.4, 400)
    X, Y = np.meshgrid(x_range, y_range)
    pressure_field = np.zeros_like(X, dtype=np.complex128)

    # Element positions and phase
    steering_rad = np.radians(steering_angle_deg)
    element_x = np.linspace(-(num_elements - 1) / 2, (num_elements - 1) / 2, num_elements) * d

    for n in range(num_elements):
        x_n = element_x[n]
        phase_n = -k * x_n * np.sin(steering_rad)
        r_n = np.sqrt((X - x_n) ** 2 + Y ** 2)
        pressure_field += (1 / r_n) * np.exp(1j * (k * r_n - omega * t + phase_n))

    # Plot
    pressure_magnitude = np.abs(pressure_field)
    pressure_magnitude /= np.max(pressure_magnitude)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(x_range * 100, y_range * 100, pressure_magnitude, shading='auto', cmap='inferno')
    plt.colorbar(label='Normalized Pressure')
    plt.title(f'Ultrasonic Interference Pattern\nN={num_elements}, Steering={steering_angle_deg}°, f={freq/1000:.0f}kHz')
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.tight_layout()
    plt.show()

# Interactive sliders
interact(
    simulate_pressure_field,
    num_elements=IntSlider(min=2, max=32, step=2, value=8, description='Elements'),
    steering_angle_deg=IntSlider(min=-60, max=60, step=5, value=30, description='Steering (°)'),
    freq=IntSlider(min=20000, max=100000, step=5000, value=40000, description='Freq (Hz)'),
    spacing_factor=FloatSlider(min=0.3, max=1.0, step=0.05, value=0.5, description='Spacing (λ)')
)
