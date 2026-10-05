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
    HORIZON = 1              # predict step t+HORIZON (try 5 or 10 if persistence is too strong)

    # ---- Sampling (see dataset.py) ----------------------------------------
    MAX_SAMPLES = 50_000     # None -> use every row
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
