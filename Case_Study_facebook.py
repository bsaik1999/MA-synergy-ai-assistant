# test_fb.py

import MAAI as train
import pandas as pd
import os
import time

if __name__ == "__main__":
    fb_file = r"C:\Users\SaiKrishna\OneDrive\Desktop\.venv\.venv\M&A\FB.csv"
    
    start = time.time()
    df, model_df, trace = train.run_mna_pipeline(fb_file, save_trace=False, train_model=False)
    
    print("\n✅ Facebook data processed and evaluated with existing model.")
    print("\nTop Synergy Deals in Facebook Data:")
    print(
        df[['Parent Company', 'Acquired Company', 'Business', 'Synergy Score']]
        .sort_values(by='Synergy Score', ascending=False)
        .head(10)
    )

    print("\nFair Value Predictions vs Actual Price:")
    print(
        model_df[['Acquired Company', 'Acquisition Price', 'Fair Value']]
        .sort_values(by='Fair Value', ascending=False)
        .head(10)
    )
    
    # Save outputs for dashboards
    os.makedirs("outputs", exist_ok=True)
    df.to_pickle("outputs/processed_fb_data.pkl")
    model_df.to_pickle("outputs/model_fb_data.pkl")
    
    print("\n✅ FB test results saved in outputs/ folder.")
    print(f"\n✅ Completed in {time.time() - start:.2f} seconds.")