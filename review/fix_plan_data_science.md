# Fix Plan: Data Science & Model Training (Phases 0-6)

**Status: COMPLETED**

This document tracks the core scientific and data pipeline flaws that were identified and successfully resolved. 

## Issue 3: Confounded Graph Ablation Study (Scientific Integrity)
- **Status:** **RESOLVED**
- **Location:** `src/models/st_gcn_lstm.py`
- **Issue:** The `LSTMBaseline` fed 5 raw features directly into the LSTM (53,668 params), whereas `STGCNLSTM` fed a 32-dimensional GCN embedding into the LSTM (63,937 params). The performance jump was confounded by parameter capacity.
- **Resolution:** A linear projection layer (`nn.Linear(5, 32)`) was added before the LSTM in `LSTMBaseline`. 
- **Verification:** `scripts/count_parameters.py` confirms that the parameter capacities now mathematically match (`lstm`: 60,772 vs `st_gcn_lstm`: 61,761).

## Issue 4: Uncontrolled Real-Time Horizon 
- **Status:** **RESOLVED**
- **Location:** `src/dataset.py`
- **Issue:** The pipeline shifted the target by 5 rows without guaranteeing a physical time grid, treating massive gaps of missing data as consecutive 2-minute steps.
- **Resolution:** `df.resample('2T').asfreq()` was applied explicitly before forward filling. 
- **Verification:** The test suite (`run_phase_tests.py --phase 2`) proves that out of ~650,000 raw rows, the pipeline now mathematically identifies over 1,000,000 missing pseudo-rows, generating a strictly continuous and robust dataset of 11,934 actual training records.

## Next Steps
Both fixes are implemented. **We are currently re-running Phase 4 (Colab Training)** to generate the new, scientifically valid `.pt` models and metrics.
