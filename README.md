# Van der Waals Equation & Mean-Field Theory

Interactive visualization of phase transitions, stability, and free energy landscapes using the Van der Waals equation of state.

## 🌐 Live Demo

Try the app online: [Streamlit Cloud Link] *(will be added after deployment)*

## 📊 Features

- **Interactive Visualization**: Explore Van der Waals isotherms and free energy landscapes
- **Phase Transitions**: Observe liquid-gas phase transitions and critical phenomena
- **Stability Analysis**: Color-coded visualization of stable, metastable, and unstable states
- **Critical Point**: One-click navigation to the critical point ($\beta_c = 2.25$, $P_c = 5.56$)

## 🚀 Running Locally

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/vdw-visualization.git
   cd vdw-visualization
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

4. Open your browser to `http://localhost:8501`

### Matplotlib Version

For a standalone matplotlib visualization with real-time sliders:

```bash
python VdW_equation_and_mean_field_theory.py
```

## 📖 Physics Background

The Van der Waals equation of state:

$$P = \frac{1}{\beta(v-b)} - \frac{a}{v^2}$$

where:
- $\beta = 1/(k_B T)$ is the inverse temperature
- $v$ is the volume per particle
- $a$ accounts for attractive interactions
- $b$ accounts for excluded volume

### Critical Point

The critical point occurs at:
- $\beta_c = \frac{27b}{8a} = 2.25$
- $P_c = \frac{a}{27b^2} = 5.56$
- $v_c = 3b = 0.03$

## 🎓 Educational Use

This visualization was created for **PHY 6536 - Statistical Mechanics** (Spring 2026) at the University of Florida.

## 📝 License

MIT License - feel free to use for educational purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
