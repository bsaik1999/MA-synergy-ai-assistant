# train_acquisitions.py

import MAAI as train

acquisitions_file = r"C:\Users\SaiKrishna\OneDrive\Desktop\.venv\.venv\M&A\acquisitions_update_2021.csv"

# Train and save trace
df, model_df, trace = train.run_mna_pipeline(acquisitions_file, save_trace=True, train_model=True)
