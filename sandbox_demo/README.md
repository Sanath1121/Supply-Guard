# Sandbox demo datasets

Four real 10-step windows from the held-out test period (17 Sep – 19 Dec 2018), saved in the dataset's
raw units, plus one invalid file. Upload them on the **Stress-Test Sandbox → Custom CSV Ingestion** tab and
press **Run Inference on Custom CSV**. The Sandbox converts raw units with the training-partition scaler
and shows a caption saying so.

Windows were chosen by a fixed rule, not by hand:

| File | Rule | Disruption story |
| --- | --- | --- |
| `1_calm_normal_operations.csv` | Test window with the median change over the next 10 minutes | Normal operations: nothing changes |
| `2_retailer_demand_shock.csv` | Largest rise of the Retailer risk index in the test period | Retail-level shock the model sees coming |
| `3_distributor_disruption.csv` | Largest Distributor rise among windows with no 0.00 Distributor reading (88 test windows contain that missing-like value) | Distributor disruption the model sees coming |
| `4_manufacturer_sudden_shock.csv` | Largest rise of the Manufacturer risk index in the test period | Sudden shock with no warning: the model misses it |
| `5_invalid_missing_total_cost.csv` | File 1 without the `Total_Cost` column | Shows input validation |

Expected Sandbox output (directed ST-GCN-LSTM, seed 42, the app's default; scaled units 0–1;
tiers = each echelon's training terciles). "Actual" is the real value 10 minutes later, which the app does not show.

| File | Supplier | Manufacturer | Distributor | Retailer | Network edges that glow | ⚡ marker |
| --- | --- | --- | --- | --- | --- | --- |
| 1 calm | 0.831 Medium | 0.762 High | 0.382 Low | 0.605 Medium | none | Manufacturer |
| 2 retailer shock | 0.839 High | 0.403 Medium | 0.461 High | **0.595 Medium** (now 0.306, actual 0.739) | none | Distributor |
| 3 distributor | 0.837 High | 0.537 Medium | **0.441 High** (now 0.070, actual 0.507) | 0.585 Medium | Distributor → Retailer | Distributor |
| 4 manufacturer | 0.831 High | **0.086 Medium** (now 0.078, actual 0.856) | 0.374 Low | 0.606 Medium | Manufacturer → Distributor | Supplier |
| 5 invalid | error: "Missing required columns: ['Total_Cost']" | | | | | |

Regenerate with `python -m scripts.make_sandbox_demo`; the values above were checked by
uploading each file in the running app.
