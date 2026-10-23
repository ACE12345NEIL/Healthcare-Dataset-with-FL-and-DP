# MediFed-DP | Healthcare Federated Learning & Differential Privacy

A clinical **Privacy-Preserving Federated Learning** system for early disease diagnosis on multi-institutional healthcare data, adhering to the architecture described in:
> **"Privacy-Preserving Federated Learning with Federated GANs for Early Heart Disease Prediction"**  
> *Vivean Arya, Nami Desai, Vihaan Bhulla, Deepa Krishnan (ICICC 2026, Springer LNNS).*

---

## 🌟 Key Features

1. **Mathematically Rigorous Differential Privacy (DP)**:
   - **Laplace Mechanism (Pure $\epsilon$-DP)**: $L_1$ update clipping sensitivity bounding with calibrated noise scale $b = C / \epsilon$.
   - **Gaussian Mechanism ($(\epsilon, \delta)$-DP)**: $L_2$ clipping with Rényi Differential Privacy (RDP) composition accounting across communication rounds.
   - Guaranteed patient confidentiality under **HIPAA** and **GDPR** standards without sharing raw EHR data.

2. **Research-Grade Model Architectures (ICICC 2026 Paper)**:
   - **Deep Residual Attention Neural Network** (Paper Section 3.3): Feature self-attention gating focusing on critical biomarkers, residual skip connections ($Z_2 + A_1$), and dropout regularization.
   - **7-Model Weighted Ensemble** (Paper Section 3.4 & Fig. 3): Multi-capacity neural networks aggregated with weights proportional to squared validation accuracy.
   - **Federated Logistic Regression & Multilayer Perceptrons (MLP)** baselines.
   - **25+ Enriched Clinical Biomarkers** (Paper Section 3.2): HOMA-IR insulin resistance proxy, metabolic risk index, nonlinear polynomial features, and ADA/AHA clinical threshold flags.

3. **Modern Clinical Web UI (Green, White, Blue)**:
   - Clean, lightweight, zero-dependency dashboard built using Python's standard library.
   - **FL + DP Training Studio**: Interactive controls for hospital nodes, communication rounds, DP mechanism, $\epsilon$, and clipping norm.
   - **Live Canvas Charts**: High-DPI round-by-round convergence curves (Accuracy & Loss) and confusion matrix heatmaps.
   - **Patient Diagnostic Tool**: Interactive clinical calculator for instant disease risk probability estimation with biometric audit and DP certificates.
   - **Paper Benchmark Comparison**: Direct comparison table with published ICICC 2026 findings.

---

## 🚀 Quick Start & Deployment

### 1. Launch the Modern Web Dashboard (Recommended)

Simply double-click `run_ui.bat` (Windows) or execute:

```bash
# Using the virtual environment python
python app.py
```
Or via the CLI:
```bash
python main.py --ui
```
Then navigate to **`http://localhost:8000`** in any web browser.

### 2. Command-Line Experiments & Pytest

```bash
# Run the complete test suite (44 unit tests covering DP, architectures & API)
python -m pytest -v

# Run full baseline benchmark suite
python main.py --stage all
```

---

## 📁 Project Structure

```
├── app.py                     # Zero-dependency Web App (Green, White, Blue UI + REST API)
├── main.py                    # CLI entrypoint with --ui, --stage options
├── run_ui.bat / run.bat       # One-click Windows launch scripts
├── run_ui.sh  / run.sh        # One-click macOS/Linux launch scripts
├── data/
│   └── diabetes.csv           # Pima Indians Healthcare Dataset (768 patients)
├── src/
│   ├── privacy.py             # Laplace Mechanism (pure ε-DP), Gaussian DP & RDP Accounting
│   ├── models.py              # Deep Residual Attention NN, 7-Model Ensemble, MLP, LogReg
│   ├── fedavg.py              # Federated Averaging with client-level DP perturbation
│   ├── preprocessing.py       # Zero-as-missing clinical cleaning & 25+ feature enrichment
│   ├── clients.py             # Disjoint hospital node partitions
│   ├── data_loader.py         # Schema validation & quality inspection
│   ├── centralized.py         # Centralized Random Forest baseline
│   ├── evaluation.py          # Clinical evaluation metrics (Acc, Prec, Rec, F1, ROC-AUC)
│   └── visualization.py       # Matplotlib evaluation plots
├── tests/                     # Comprehensive test suite (44 tests)
│   ├── test_app.py            # Web UI endpoint verification
│   ├── test_paper_and_laplace_dp.py # Laplace DP & Deep Attention ResNet gradient tests
│   ├── test_privacy.py        # Noise & clipping sensitivity tests
│   ├── test_federated.py      # FedAvg convergence tests
│   └── test_preprocessing.py  # Clinical data cleaning tests
└── results/                   # Benchmark CSVs, JSON runs, and plots
```

---

## 📊 Differential Privacy Mathematics

### 1. Laplace Mechanism (Pure $\epsilon$-DP)
For client update $\Delta w_k$, sensitivity is bounded by clipping parameter $C$:
$$\| \Delta w_k \|_1 \le C$$
Pure $\epsilon$-differential privacy is guaranteed by adding noise drawn from the Laplace distribution:
$$\eta \sim \text{Laplace}\left(0, \frac{C}{\epsilon_{\text{round}}}\right)$$

### 2. Composition across Communication Rounds
Across $T$ communication rounds, each client update maintains privacy:
- **Basic Composition**: $\epsilon_{\text{total}} = T \cdot \epsilon_{\text{round}}$
- **Advanced Composition (Dwork et al.)**:
$$\epsilon_{\text{total}} = \sqrt{2 T \ln(1/\delta')} \epsilon_{\text{round}} + T \epsilon_{\text{round}} (e^{\epsilon_{\text{round}}} - 1)$$
With target $\epsilon = 1.0$ (as demonstrated in Arya et al. ICICC 2026), the framework prevents re-identification attacks and membership inference while preserving clinical diagnostic accuracy.
"# Healthcare-Dataset-with-FL-and-DP" 
