# MiniFold: LLM-Driven Lightweight Protein Folder

MiniFold is an ultra-lightweight protein structure prediction system that bridges the gap between Large Language Models (LLMs) and rigorous biophysics. It leverages LLMs (like Ark/Doubao) for "intuition-based" secondary structure prediction and refinement, then uses a specialized iGPU-accelerated differentiable physics engine to fold these predictions into geometrically valid 3D structures.

> **Key Philosophy:** "LLM proposes, Physics disposes."

## Core Features

### 1. LLM-Driven Architecture
- **Multi-Model Voting**: Aggregates predictions from multiple biological LLMs to generate high-confidence Secondary Structure (SS) candidates.
- **Environment-Aware Refinement**: Uses LLMs to refine structures based on natural language descriptions (e.g., "membrane protein", "high temperature"), dynamically adjusting constraints.
- **Strict Logic Control**: Enforces user constraints (e.g., chain count) via prompt engineering and algorithmic post-processing.

### 2. Physically Valid Folding (iGPU Engine)
Unlike simple geometric builders, MiniFold's `igpu_predictor` is a differentiable physics engine that ensures:
- **True Peptide Geometry**: 
  - Locked $\omega$ angles (Trans-peptide, 180°).
  - Correct Carbonyl Oxygen placement (planar & trans to N).
  - Correct L-amino acid chirality.
- **Ramachandran Compliance**: Discrete sampling from allowed $\phi/\psi$ basins ($\alpha$, $\beta$, PPII), avoiding forbidden regions.
- **Steric Clash Resolution**: Hard-sphere repulsion (< 2.0Å) prevents atom overlap.
- **Chain Continuity**: Strong penalties for $C\alpha-C\alpha$ bond stretching/compression (~3.8Å).

### 3. Hardware Acceleration
- **DirectML Support**: Runs natively on Windows iGPUs (Intel Iris/Arc, AMD Radeon, NVIDIA) via `torch-directml`.
- **Cross-Platform**: Fallback support for CUDA (NVIDIA), MPS (Mac), and CPU.

### 4. Interactive Web UI
- **Real-time Visualization**: Integrated 3Dmol.js viewer showing folding trajectories.
- **Live Progress**: Synchronized progress bars tracking every step from LLM query to physics optimization.
- **Environment Management**: Native support for Conda environments to isolate dependencies.

## Installation

### Prerequisites
- Python 3.9+
- Conda (Recommended)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MiniFold.git
   cd MiniFold
   ```

2. Create a Conda environment:
   ```bash
   conda create -n minifold python=3.10
   conda activate minifold
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   *Note: For Windows iGPU acceleration, ensure `torch-directml` is installed correctly.*

4. **Configure Environment**
   Create a `.env` file in the project root with your API Keys (optional if using UI):
   ```ini
   # Volcengine (Ark)
   ARK_API_KEY=your_api_key_here
   ARK_MODEL=doubao-pro-32k
   ```

## Usage

1. Start the Web Server:
   ```bash
   python web_ui/server.py
   ```

2. Open Browser:
   Navigate to `http://localhost:5000`.

3. Run a Job:
   - **Input**: Paste a protein sequence (FASTA).
   - **Environment**: Describe the target environment (e.g., "cytosolic, stable beta-sheet core").
   - **Settings**: Adjust chain count (1 for single domain) or iteration steps.
   - **Click "Run"**: Watch the LLM think and the physics engine fold.

## Technical Details

### The Pipeline
1. **Sequence Analysis**: PyBioMed & LLMs predict initial SS (H/E/C).
2. **Consensus Voting**: Multiple LLM calls vote on the best SS topology.
3. **Refinement Loop**: Top candidate is refined by LLM with environmental context.
4. **Physics Folding**:
   - **Initialization**: NeRF (Natural Extension Reference Frame) places atoms.
   - **Optimization**: PyTorch-based gradient descent minimizes a composite energy function:
     $$L = w_{ss}L_{ss} + w_{rama}L_{rama} + w_{clash}L_{clash} + w_{bond}L_{bond} + \dots$$
5. **Output**: Standard PDB file ready for PyMOL/Chimera.

## License
MIT License
