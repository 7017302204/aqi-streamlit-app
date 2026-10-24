# Delhi next-hour AQI forecast (Streamlit)

Forecasts the CPCB Air Quality Index one hour ahead for Delhi stations, using the LightGBM model from the study
(159 engineered features). Research prototype, not for health or regulatory decisions.

## 1. Create the model files (once)
In the project notebook run the cell **STEP 44**. It writes `aqi_pipeline.py` and an `artifacts/` folder, checks that the
app's feature pipeline reproduces the notebook's features (look for `PARITY OK`), and downloads
`aqi_streamlit_artifacts.zip`.

Unzip it into this folder so you have:

```
aqi_streamlit_app/
  app.py
  aqi_pipeline.py            <- overwritten by the one from the notebook (identical)
  requirements.txt
  artifacts/
    model_lgbm.joblib
    feature_cols.json
    station_mapping.json
    history_clean.csv.gz
    metrics.json
```

## 2. Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## 3. Deploy (Streamlit Community Cloud)
1. Push this folder to a GitHub repository (all files above, including `artifacts/`; the files are a few MB).
2. On share.streamlit.io choose **New app**, select the repository, branch and `app.py`.
3. Deploy. Add `streamlit>=1.50` and the other packages through `requirements.txt` (already included).

Any host that runs Python works (Hugging Face Spaces with the Streamlit SDK, a VM, Docker).

## Using the app
* **Replay historical data:** pick a station and an hour; the app forecasts the next hour and shows the actual value.
* **Upload recent data:** CSV with columns `Station, timestamp, PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, Ozone, Benzene,
  AT, BP, SR, RH, WS, WD`, at least **72 consecutive hourly rows per station** without gaps or missing values (the
  24 h averages and the chained lags need that history). The download button in the app gives an example file.

## Notes
* The deployed model is LightGBM because its RMSE is statistically indistinguishable from the Random Forest's in the
  study and it is small and fast. The Random Forest is large (hundreds of MB), which exceeds GitHub's file limit.
* The model is trained on the training split only, so the metrics in the app match the paper. To refit on all data,
  change the fit in STEP 44 to use train + validation (then the reported test metrics no longer apply).
* Forecasts are least reliable in the post-monsoon and winter seasons (validated only on late July – September 2019),
  at very high AQI, and for stations outside the six used in training.
