# AI Agent Instructions: Projet Santé

## Project Overview
This is a **clinical predictive modeling project** focused on Primary Biliary Cirrhosis (PBC) patient data using LSTM neural networks for risk estimation over time. The project processes longitudinal clinical data, performs statistical analysis, and trains deep learning models for time-discretized risk prediction.

## Architecture & Data Flow

### Core Pipeline
1. **Data Cleaning** (`src/correct_data.py`) → Processed CSV in `data/`
2. **Data Analysis** (`src/process_data.py`) → Visualizations & statistics to `data/` (PNG files)
3. **Model Training** (`src/model.py`) → LSTM-based risk estimator (incomplete, needs forward pass completion)

### Key Data Transformations
- **Input**: Raw CSV with patient IDs, multiple visits per patient, clinical measurements (`ser_bilir`, `serChol`, `albumin`, `alkaline`, `SGOT`, `platelets`, `prothrombin`, `histologic`)
- **Cleaning steps** (chained in `correct_data.main()`):
  - Remove `total_protein` column
  - Fix unit error for patient ID 104 (ser_bilir values ÷100)
  - Remove `date_diag` column
  - Impute continuous vars with median, categorical with mode
  - Remap labels: 2→0 (for binary/3-class outcome prediction)
- **Analysis**: Correlation matrices, time-delta distributions, survival time distributions by event status

### Directory Structure
```
src/config.py          # Centralized path management (PROJECT_ROOT, DATA_DIR, RESULTS_DIR)
src/correct_data.py    # Data cleaning pipeline with modular functions
src/process_data.py    # EDA with matplotlib/seaborn visualizations
src/model.py           # LSTM_risk_estimator class (PyTorch, batch_first=True)
data/                  # Raw input: clinical_data_pbc.csv, outputs: *_cleaned.csv + PNG plots
results/               # Reserved for model outputs (currently unused)
```

## Developer Workflows

### Running Data Processing (Windows PowerShell with uv)
```powershell
# Activate venv (if using venv; uv can skip this)
& .venv\Scripts\Activate.ps1

# Clean raw data
uv run python -m src.correct_data

# Generate visualizations & statistics
uv run python -m src.process_data
```

### Key Commands
- Each module has `if __name__ == "__main__"` entry point
- All modules start with `print("RUNNING:", Path(__file__).resolve())` for debugging
- Use `uv run` for consistent environment (project uses uv package manager, Python 3.13+)

## Project-Specific Patterns

### 1. Modular Data Transformation
Functions in `correct_data.py` are **single-responsibility and chainable**:
- Each function accepts DataFrame, returns modified DataFrame
- Chain them in `main()`: `df = remove_protein(df)` → `df = correct_ser_bilir_unit_104(df)` → ...
- Use `errors="ignore"` when dropping optional columns (handles missing columns gracefully)

### 2. Configuration via Centralized Paths
- **Never hardcode paths**: Import from `config.py`
- `DATA_DIR = PROJECT_ROOT / "data"` uses pathlib (not strings)
- Directories auto-create with `mkdir(parents=True, exist_ok=True)`

### 3. Visualization Outputs to Data Directory
- EDA plots saved as PNG to `DATA_DIR / f"descriptive_name_{variable}.png"`
- Close figures with `plt.close()` to prevent memory bloat
- Use consistent style: seaborn for statistical plots, matplotlib subplots for comparisons

### 4. Handling Patient Longitudinal Data
- Clinical data is **multi-visit per patient** (identified by `id` column)
- Group operations: `groupby('id')` for patient-level aggregations
- Sort by `['id', 'times']` before time-delta calculations
- Last observation per patient: `.groupby('id').tail(1)` for survival status

### 5. LSTM Model Design Pattern (Incomplete)
- **Input**: Sequence of clinical measurements (batch_first=True)
- **Output**: Time-discretized risk predictions (softmax over `number_time_discrete` bins)
- Architecture: LSTM → Linear layer → Softmax
- **TODO**: Complete `forward()` method with LSTM output processing and final classification

## Dependencies & Environment

### Critical Libraries
- **PyTorch 2.9.1+**: Neural network framework (LSTM, Linear layers)
- **Pandas 2.3.3+**: Data manipulation (read_csv, fillna, groupby)
- **NumPy 2.4.1+**: Numerical ops (log1p transformation, correlation matrices)
- **Seaborn/Matplotlib**: Visualization (heatmaps, histplots, bar charts)
- **Python 3.13+**: Required version (strict in pyproject.toml)

### Environment Management
- Uses `uv` for fast dependency resolution (see `uv.lock`)
- Virtual environment in `.venv/`
- All code assumes relative imports from `src/` package (e.g., `from .config import DATA_DIR`)

## Common Pitfalls & Conventions

1. **Import Strategy**: Use relative imports within `src/` package (e.g., `from .config import DATA_DIR`), not absolute paths
2. **Missing Data Handling**: Median for continuous variables, mode for categorical—check column existence before imputation
3. **Label Encoding**: Event outcomes are mapped (0=Censored/Alive, 1=Transplanted, 2=Deceased initially) → transformed to binary (2→0)
4. **DataFrame Copies**: Use `.copy()` to avoid SettingWithCopyWarning (seen in `process_data.py`)
5. **Visualization**: Always close figures in loops to prevent memory issues; use `plt.figure()` for new plots

## Integration Points for Extensions

- **Model completion**: Finish `LSTM_risk_estimator.forward()` to return logits or probabilities
- **Training pipeline**: Implement data loading (create DataLoader from cleaned CSV), train/val splits, loss function (cross-entropy for classification)
- **Results directory**: Currently unused; designate for model checkpoints, predictions, and metrics
- **French dataset context**: Variable names are French (e.g., `serBilir`, `albumin`); maintain consistency in documentation
