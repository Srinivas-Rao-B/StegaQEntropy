# StegaQEntropy

### An Entropy-Driven Adaptive Reversible Image Steganography Framework with Multimodal Surrogate Learning and Quantum-Assisted Region Selection and Security

**StegaQEntropy** is a hybrid classical–quantum reversible image steganography framework that integrates image-statistical intelligence, single-level `db2` discrete wavelet transform (DWT) analysis, multimodal surrogate modeling (**SQE-Net**, **QOQA**, and **ACER**), physical quantum random number generation (**QRNG** on IBM Quantum `ibm_fez`), and hierarchical **BB84 QKD** session coordination.

To preserve **exact, bit-for-bit cover reversibility** ($\text{BER} = 0$, $\text{MSE}_{\text{recon}} = 0$) without transform-domain rounding errors, DWT subbands are used strictly as an analysis domain while secret data is embedded directly in the **8-bit integer pixel domain** via ML-guided reversible **Prediction-Error Expansion (PEE)**. The QOQA-selected region map is encrypted with the QKD master key ($K_M$) and hidden inside key-seeded reserved pixels, requiring no separate mask or reversal metadata over public channels.

---

# 1. Framework Overview

The StegaQEntropy framework follows the following overall pipeline:

```text
                         Cover Image (8-bit Grayscale)
                                      │
                                      ▼
                       ┌─────────────────────────────┐
                       │ Image Acquisition &         │
                       │ PEE Capacity Estimation     │
                       └──────────────┬──────────────┘
                                      │
                       ┌──────────────┴──────────────┐
                       ▼                             ▼
        ┌──────────────────────────┐   ┌──────────────────────────┐
        │ Entropy & Image          │   │ db2 DWT Decomposition    │
        │ Intelligence Analysis    │   │ {LL, LH, HL, HH}         │
        │ (Analysis Domain Only)   │   │ (Analysis Domain Only)   │
        └─────────────┬────────────┘   └─────────────┬────────────┘
                      └──────────────┬───────────────┘
                                     ▼
                       ┌─────────────────────────────┐
                       │ Multimodal Surrogate &      │
                       │ SQE-Net (64-Qubit Regional) │
                       └──────────────┬──────────────┘
                                      │
                                      ▼
                       ┌─────────────────────────────┐
                       │ QOQA / QAOA Region          │
                       │ Selection (E_8, K=275)      │
                       └──────────────┬──────────────┘
                                      │
                                      ▼
                       ┌─────────────────────────────┐
                       │ Classical Q/K/V Attention   │
                       │ + ACER (32-Qubit Global)    │
                       └──────────────┬──────────────┘
                                      │
                                      ▼
                       ┌─────────────────────────────┐
                       │ IBM QRNG (ibm_fez) +        │
                       │ Hierarchical BB84 QKD Keys  │
                       └──────────────┬──────────────┘
                                      │
                                      ▼
                       ┌─────────────────────────────┐
                       │ Integer Pixel-Domain PEE    │
                       │ (Payload + Hidden ML Map)   │
                       └──────────────┬──────────────┘
                                      │
                                      ▼
                         Stego Image (8-bit Integer)
                                      │
                                      ▼
                       ┌─────────────────────────────┐
                       │ Receiver Extraction &       │
                       │ Bit-Exact Cover Restoration │
                       └─────────────────────────────┘
```

### Reproducibility and First Run

Generated output files and intermediate artifacts can be regenerated from scratch by executing the pipeline via `main.py` followed by `src/adaptive_embedding.py`. Before running, update any local input/output paths in `ml_module/ml_configuration.py` and `src/` to match the local directory of the cloned repository (`Cover_images/`, `Secret_Images/`, and `output/`).

### Payload Configurations

* **Consolidated 20,000-Bit Benchmark (`Secret_Images/TextData.txt`):** The primary evaluation fixes a consolidated budget of $M_{\text{total}} = 20,000$ bits per carrier, encompassing the 64-bit synchronization header, the encrypted secret text payload ($19,664$ bits for $512\times 512$; $19,536$ bits for $512\times 768$), and the internalized encrypted ML region map ($272$ bits for $512\times 512$; $400$ bits for $512\times 768$).
* **High-Payload RGB Secret-Image Mode ($41,065$ bits):** For image-hiding experiments, an input RGB secret image is resized to $128\times 128$ pixels and JPEG-compressed with adaptive quality selection to satisfy the target payload budget ($40,793$-bit protected stream + $272$-bit ML map). The receiver extracts the compressed bitstream and reconstructs the secret image without needing the original secret image file.

---

# 2. Repository Structure

```text
StegaQEntropy/
│
├── main.py
├── requirements.txt
├── README.md
│
├── Cover_images/
│   ├── lena512.jpg
│   ├── barbara512.jpg
│   ├── lake512.jpg
│   ├── boat512.jpg
│   ├── peppers512.jpg
│   └── A16.jpg
│
├── Secret_Images/
│   └── TextData.txt
│
├── src/
│   ├── image_acquisition.py
│   ├── capacity_estimation.py
│   ├── dwt_decomposition.py
│   ├── entropy_intelligence_engine.py
│   └── adaptive_embedding.py
│
├── ml_module/
│   ├── dataset_builder.py
│   ├── dataset_cleaner.py
│   ├── gaussian_probability_plot.py
│   ├── ml_configuration.py
│   ├── ml_random_manager.py
│   ├── post_embedding_analysis_fixed.py
│   ├── surrogate_model.py
│   └── output/
│
├── qrng/
│   ├── backend_manager.py
│   ├── ibm_connection.py
│   ├── ibm_test.py
│   ├── qrng_pool.py
│   ├── qrng_randomness_engine.py
│   ├── qrng_summary.py
│   ├── quantum_circuit.py
│   ├── quantum_execution.py
│   ├── randomness_tests.py
│   ├── seed_manager.py
│   ├── state_analysis.py
│   ├── transpiler.py
│   └── output/
│
└── output/
    ├── adaptive_embedding/
    ├── dwt_decomposition/
    ├── message_preparation/
    ├── output_modules/
    └── quantum_key_hierarchy/
```

---

# 3. Core Implementation

## 3.1 Image Acquisition and Capacity Estimation

These stages load the 8-bit grayscale cover image, extract global statistical and GLCM texture descriptors (entropy, variance, gradient, contrast, homogeneity, energy), and compute theoretical ($C_{\text{th}} = MN$) and usable integer pixel-domain PEE capacities ($C_{\text{pee}}$).

* **Scripts:** `src/image_acquisition.py`, `src/capacity_estimation.py`
* **Execution Artifacts (`output/output_modules/`):** `image_acquisition_Execution.txt`, `capacity_estimation_Execution.txt`

```bash
python src/image_acquisition.py
python src/capacity_estimation.py
```

---

# 4. DWT Decomposition (Analysis Domain Only)

Performs a single-level two-dimensional Daubechies-2 (`db2`) Discrete Wavelet Transform producing `{LL, LH, HL, HH}` subbands. These subbands supply spectral patches ($16\times 16\times 4$) and regional subband energies to the surrogate model, while **no embedding modification is performed in the wavelet domain** to avoid floating-point rounding errors.

* **Script:** `src/dwt_decomposition.py`
* **Artifacts (`output/dwt_decomposition/`):** `LL.npy`, `LH.npy`, `HL.npy`, `HH.npy`, `LL_normalized.npy`, `LH_normalized.npy`, `HL_normalized.npy`, `HH_normalized.npy`, `dwt_metadata.json`, `dwt_coefficients.json`, `candidate_regions.json`, `adaptive_regions.json`, `band_priority.json`, `coefficient_statistics.json`

```bash
python src/dwt_decomposition.py
```

---

# 5. Entropy Intelligence and Regional Analysis

Extracts multi-domain regional descriptors across candidate $32\times 32$ blocks—including Shannon entropy, relative entropy, entropy gradient, GLCM texture measures, and graph centrality—and classifies blocks into `PRIMARY`, `SECONDARY`, `TERTIARY`, and `REJECTED` tiers.

* **Script:** `src/entropy_intelligence_engine.py`
* **Execution Artifact (`output/output_modules/`):** `entropy_intelligence_engine_Execution.txt`

```bash
python src/entropy_intelligence_engine.py
```

---

# 6. SQE-Net / Multimodal Surrogate Modeling

Constructs multimodal regional representations ($32\times 32$ raw patches, $16\times 16\times 4$ `db2` DWT patches, and regional intelligence vectors) and runs controlled post-embedding 8-neighbour stress analysis ($P_{\text{safe}}, P_{\text{risk}}$) alongside Gaussian PDF/CDF probability modeling. **SQE-Net** processes the fused features using **64 logical qubits** (eight independent 8-qubit registers simulated via $2^8 = 256$ complex amplitudes with $R_Y/R_Z$ rotations, Hadamard gates, and cyclic CNOT rings) to predict seven regional suitability objectives.

* **Scripts (`ml_module/`):** `dataset_builder.py`, `dataset_cleaner.py`, `post_embedding_analysis_fixed.py`, `gaussian_probability_plot.py`, `surrogate_model.py`, `ml_configuration.py`, `ml_random_manager.py`
* **Artifacts (`ml_module/output/`):**
  * Datasets: `master_dataset.csv`, `master_dataset.json`, `master_dataset.npy`, `dataset_manifest.json`, `dataset_statistics.json`
  * Post-embedding analysis: `actual_post_embedding_complete_dataset.csv`, `actual_post_embedding_dataset.csv`, `actual_post_embedding_final_dataset.csv`, `actual_post_embedding_ml_dataset.csv`, `actual_post_embedding_ml_targets.csv`, `actual_post_prediction_targets.csv`, `actual_post_region_ranking.csv`, `actual_region_prediction_summary.csv`, `actual_safe_capacity.csv`, `final_embedding_candidates.json`, `final_region_embedding_decisions.csv`, `region_priority_groups.json`

```bash
python ml_module/surrogate_model.py
```

---

# 7. QOQA / QAOA Optimization

**QOQA** formulates regional selection as a Quadratic Unconstrained Binary Optimization (QUBO) problem mapped to an Ising Hamiltonian, enforcing strict 8-neighbour spatial exclusion ($E_8$) and a target quota of $K = 275$ regions. It executes depth-1 QAOA ($P = 1$, $N_{\text{shots}} = 1024$) across partitioned sub-blocks ($m_b \le 8$) via state-vector simulation to select conflict-free, distortion-tolerant candidate regions.

* **Visualization Artifacts (`ml_module/output/`):** `02_sqe_net_internal_architecture.png`, `03_sqe_net_8_qubit_circuit.png`, `06_qoqa_corrected_pipeline.png`, `07_qoqa_qubo_formulation.png`, `08_qoqa_qubo_to_ising.png`, `10_qoqa_qaoa_style_block_circuit.png`

---

# 8. ACER Global Quality Estimation

Following QOQA selection, selected regional embeddings are aggregated via classical masked scaled dot-product $Q/K/V$ attention and passed into **ACER**, which uses **32 logical qubits** (four independent 8-qubit registers with cyclic CNOT entanglement) to predict holistic carrier PSNR prior to pixel expansion.

* **Visualization Artifacts (`ml_module/output/`):** `09_region_latent_acer_representation.png`, `10_acer_angle_encoding.png`, `11_full_32_qubit_acer_circuit.png`, `27_acer_qoqa_circuit_description.txt`

---

# 9. QRNG & NIST SP 800-22 Verification

Acquires physical quantum randomness from the 156-qubit IBM Quantum processor (`ibm_fez`) using 32 physical qubits executing Hadamard superposition and computational-basis measurement ($6 \text{ blocks} \times 16,384 \text{ shots} = 3,145,728$ bits). The generated bitstream is validated across all 15 NIST SP 800-22 statistical test families at $\alpha = 0.01$ and stored in a verified seed pool.

* **Scripts (`qrng/`):** `backend_manager.py`, `ibm_connection.py`, `ibm_test.py`, `qrng_pool.py`, `qrng_randomness_engine.py`, `qrng_summary.py`, `quantum_circuit.py`, `quantum_execution.py`, `randomness_tests.py`, `seed_manager.py`, `state_analysis.py`, `transpiler.py`
* **Artifacts (`qrng/output/`):** `qrng_pool.bin`, `qrng_pool.txt`, `qrng_pool_history.txt`, `qrng_pool_summary.txt`, `qrng_statistics.txt`, `randomness_report.txt`, `qrng_final_report.txt`, `backend_information.txt`, plus statevector/Bloch/circuit diagnostic plots.

---

# 10. Seed and History Verification

Maintains tamper-evident module seed records derived from the verified `ibm_fez` QRNG bitstream so experimental runs can be audited and reproduced. Authoritative seed files under `final_verified_seeds/` should be read rather than overwritten.

* **Artifacts (`qrng/output/history_check/`):** `history_database.txt`, `history_summary.txt`, `final_verified_seeds/` (`dwt_seed_verified.txt`, `entropy_seed_verified.txt`, `qkd_seed_verified.txt`, `region_seed_verified.txt`, `adaptive_embedding_seed_verified.txt`, `extraction_seed_verified.txt`, `verification_seed_verified.txt`), `regenerated_seeds/`, `verified_seeds/`

---

# 11. Hierarchical Quantum Key Management

Simulates BB84 QKD sessions using physical QRNG entropy (with a 10% sacrificial QBER check and HMAC confirmation) to establish a 128-bit master key ($K_M$) and 32 independent 63-bit chunk subkeys ($K_j$). $K_M$ masks the global stream, seeds tie-breaking pixel ordering, locates reserved pixels, and encrypts the hidden ML map, while subkeys protect individual payload chunks and are never embedded.

* **Artifacts (`output/quantum_key_hierarchy/`):** `chunk_key_mapping.json`, `dependency_report.json`, `embedding_readiness.json`, `hierarchy.json`, `hierarchy_report.txt`, `key_configuration.json`, `manifest.json`, `qkd_package.json`, `qkd_plan.json`, `readiness.json`, `security_analysis.json`, `statistics.json`

> **Simulation Disclosure:** BB84 QKD sessions and quantum circuit stages (SQE-Net, QOQA, ACER) are executed via classical Qiskit Aer/state-vector simulation; the reported $\text{QBER} = 0$ corresponds to an ideal noiseless simulation without an eavesdropper rather than a physical optical channel.

---

# 12. Adaptive Integer Pixel-Domain Embedding and Extraction

Embeds secret data directly in the **8-bit integer pixel domain** via two-layer checkerboard reversible Prediction-Error Expansion (PEE, $\max \vert{}\Delta\vert{} \le 1$), scaling roughness by $\lambda_{\text{ML}} = 0.75$ inside QOQA-selected regions and breaking ties via $K_M$. The encrypted ML region bitmap ($272$ bits for $512\times 512$; $400$ bits for $512\times 768$) is embedded into a disjoint $K_M$-seeded reserved pixel set, enabling exact payload recovery and lossless cover restoration ($\text{BER} = 0$, $\text{MSE}_{\text{recon}} = 0$) with zero side-channel metadata.

* **Script:** `src/adaptive_embedding.py`
* **Artifacts (`output/adaptive_embedding/`):** `adaptive_embedding_metrics.json`, `adaptive_embedding_plan.json`, `adaptive_stego.png` / `.npy`, `extracted_message.bin`, `selected_embedding_regions.png`

**Single-Terminal Run:**
```bash
python src/adaptive_embedding.py
```

**Manual Two-Terminal Sender / Receiver Execution:**
```bash
# Terminal 1 — Receiver
python src/adaptive_embedding.py receiver

# Terminal 2 — Sender
python src/adaptive_embedding.py sender
```

---

# 13. Benchmark Cover Images

Evaluations use six standard $512 \times 512$ 8-bit grayscale USC-SIPI benchmark images (`lena512.jpg`, `barbara512.jpg`, `lake512.jpg`, `boat512.jpg`, `peppers512.jpg`, `A16.jpg`) stored in `Cover_images/`, along with 19 grayscale-converted $512 \times 768$ Kodak images (`kodim01–07`, `09–12`, `14–17`, `19`, `21–23`). Standard USC-SIPI images are available from the [USC-SIPI Image Database](https://sipi.usc.edu/database/).

---

# 14. Execution Traceability Logs

Module-level execution records are saved under `output/output_modules/` for full pipeline traceability:

* `image_acquisition_Execution.txt`
* `capacity_estimation_Execution.txt`
* `qrng_randomness_engine_Execution.txt`
* `history_checker_Execution.txt`
* `message_preparation_Execution.txt`
* `quantum_key_hierarchy_Execution.txt`
* `dwt_decomposition_Execution.txt`
* `entropy_intelligence_engine_Execution.txt`
* `dataset_builder_Execution.txt`
* `post_embedding_analysis_fixed_Execution.txt`
* `gaussian_probability_plot_Execution.txt`
* `surrogate_model_Execution.txt`

---

# 15. Computational Environment & Execution Workflow

### Step 1 — Install Dependencies
```bash
git clone [https://github.com/Srinivas-Rao-B/StegaQEntropy.git](https://github.com/Srinivas-Rao-B/StegaQEntropy.git)
cd StegaQEntropy
pip install -r requirements.txt
```

### Step 2 — Run the Master Pipeline (`main.py`)
Run `main.py` and select **`0. Run Complete Pipeline`** (or execute stages `1` through `11` sequentially) to generate all intermediate features, surrogate predictions, and QKD key hierarchies:

```bash
python main.py
```

```text
======================================================================
                  STEGAQENTROPY
     HYBRID QUANTUM STEGANOGRAPHY SYSTEM
======================================================================

======================================================================
SELECT EXECUTION MODE
======================================================================
0. Run Complete Pipeline
1. Image Acquisition & Capacity Estimation
2. QRNG Randomness Engine
3. History Checker
4. Message Preparation
5. Quantum Key Hierarchy
6. DWT Decomposition
7. Entropy Intelligence Engine
8. Dataset Builder
9. Post Embedding Analysis
10. Gaussian Probability Plot
11. Surrogate Model
```

### Step 3 — Embed, Transmit, and Verify Extraction
Execute `src/adaptive_embedding.py` (single-terminal or sender/receiver mode) and inspect `output/adaptive_embedding/adaptive_embedding_metrics.json` to verify zero payload bit errors ($\text{BER} = 0$) and bit-exact cover restoration ($\text{MSE}_{\text{recon}} = 0$).
