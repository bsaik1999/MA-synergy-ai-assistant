from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import io
import MAAI as train  # your M&A pipeline module

app = FastAPI(title="M&A Synergy Engine API")

@app.get("/healthcheck")
def healthcheck():
    return {"status": "running"}

@app.post("/process_acquisitions")
async def process_acquisitions(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    # Save temp CSV to disk if your pipeline expects a file path
    temp_path = "temp_uploaded_acquisition.csv"
    df.to_csv(temp_path, index=False)

    processed_df, model_df, _ = train.run_mna_pipeline(temp_path, save_trace=False, train_model=False)

    top_synergy = (
    processed_df[['Parent Company', 'Acquired Company', 'Business', 'Synergy Score']]
    .sort_values(['Parent Company', 'Synergy Score'], ascending=[True, False])
    .groupby('Parent Company')
    .head(3)   # top 3 deals per parent company, adjust number as you want
)
    fair_value_preds =( model_df[['Acquired Company', 'Acquisition Price', 'Simulated Fair Value Mean', 'Simulated Fair Value VaR_5', 'Simulated Fair Value VaR_95', 'P(Fair Value > Price)']]\
                .sort_values(by='Simulated Fair Value Mean', ascending=False).head(5)
)

    # Convert to JSON-serializable dicts
    return JSONResponse(content={
        "top_synergy_deals": top_synergy.to_dict(orient="records"),
        "fair_value_predictions": fair_value_preds.to_dict(orient="records")
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
