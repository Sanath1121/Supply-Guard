"""All hyperparameters and column mappings in one place."""


class Config:
    # ---- Paths -------------------------------------------------------------
    RAW_DATA_PATH = "data/raw/SCRM_timeSeries_2018_train.csv"
    SCALER_PATH = "outputs/models/scaler.joblib"
    CKPT_DIR = "outputs/models"

    # ---- Topology ----------------------------------------------------------
    NUM_NODES = 4
    NODE_NAMES = ["Supplier", "Manufacturer", "Distributor", "Retailer"]
    # Columns are ALWAYS selected by name (raw CSV order is S, D, M, R!)
    DATE_COL = "Timestamp"
    NODE_TARGET_COLS = ["RI_Supplier1", "RI_Manufacturer1",
                        "RI_Distributor1", "RI_Retailer1"]
    COST_COL = "Total_Cost"
    FEATURE_COLS = NODE_TARGET_COLS + [COST_COL]   # first 4 = nodes, last = cost
    NUM_INPUT_FEATURES = 5
    NODE_FEAT_DIM = 2        # per node per step: [its own RI, Total_Cost]

    # ---- Forecasting -------------------------------------------------------
    SEQ_LEN = 10             # window [t-L+1 .. t]
    HORIZON_MIN = 10         # nominal prediction horizon in minutes
    CADENCE_MIN = 2.0        # nominal sampling cadence in minutes
    HORIZON = 5              # predict step t+HORIZON (5 steps: nominal 10 min, median 10.0 min, mean 10.8 min)
    GAP_MAX_MIN = 6.0        # max allowable time gap in minutes before segment split (3x cadence)
    FFILL_LIMIT = 5          # max consecutive null rows to forward fill
    TIMESTAMP_FORMAT = "%m/%d/%Y %I:%M:%S %p"
    DATE_FORMAT = TIMESTAMP_FORMAT  # alias for backward-compatibility

    # ---- Sampling (see dataset.py) ----------------------------------------
    MAX_SAMPLES = None       # None -> use every row
    SAMPLING = "stride"      # "stride": evenly spaced over the whole span | "head": first N rows
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.80, 0.10, 0.10

    # ---- Architecture ------------------------------------------------------
    GCN_HIDDEN_DIM = 32
    LSTM_HIDDEN_DIM = 64
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.2
    HEAD_HIDDEN = 32
    FC_DROPOUT = 0.2
    GRAPH_MODE = "directed"  # "directed" (S->M->D->R with separate up/down weights) | "symmetric" (Kipf)
    RESIDUAL = True          # predict y_t + delta, so the model starts at the persistence baseline

    # ---- Training ----------------------------------------------------------
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    MAX_EPOCHS = 50
    PATIENCE = 10
    GRAD_CLIP = 1.0
    SEEDS = [42, 43, 44, 45, 46]   # report mean +/- std over these
