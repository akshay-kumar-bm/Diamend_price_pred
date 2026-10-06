# Diamond (Gemstone) Price Prediction

Regression project that predicts diamond price from carat, dimensions and the 4Cs-style categorical attributes, structured as a modular ML pipeline with a Flask form.

## Dataset
`notebooks/data/gemstone.csv` (~10 MB; columns: id, carat, cut, color, clarity, depth, table, x, y, z, price).

## Approach
- EDA in `notebooks/EDA.ipynb`.
- Preprocessing (notebook): median imputation + StandardScaler for numerics; most-frequent imputation + OrdinalEncoder (cut: Fair..Ideal, color: D..J, clarity: I1..IF) + scaling for categoricals; 70/30 split.
- Models compared: LinearRegression, Lasso, Ridge, ElasticNet, DecisionTreeRegressor.
- Results printed in `notebooks/Model Training.ipynb` (test set; "R2 score" printed as percentage):

| Model | RMSE | MAE | R2 |
|---|---|---|---|
| LinearRegression | 1013.9 | 674.0 | 93.69 |
| Lasso | 1013.9 | 675.1 | 93.69 |
| Ridge | 1013.9 | 674.1 | 93.69 |
| ElasticNet | 1533.4 | 1060.7 | 85.56 |
| DecisionTree | 838.7 | 424.4 | 95.68 |

## Tech stack
Python, pandas, numpy, seaborn, scikit-learn, Flask.

## Structure
```
application.py              # Flask app (/, /predict form)
src/components/             # data_ingestion, data_transformation, model_trainer
src/pipelines/              # training_pipeline, prediction_pipeline
src/{logger,exception,utils}.py
notebooks/                  # EDA, Model Training, data/gemstone.csv
templates/                  # index.html, form.html
artifacts/, logs/           # generated outputs (committed)
```

## Setup
```bash
pip install -r requirements.txt     # includes -e . (setup.py)
python src/pipelines/training_pipeline.py
python application.py
```

## Limitations / future work
- The pipeline code is inconsistent: the trainer saves the best model's name (string) instead of the model object, and `application.py` calls `Predict.Predict` while the class defines `predict`; `prediction_pipeline.py` uses `os` without import and imports `scr.exception`. Serving likely fails without fixes.
- Results are from the notebook only; no hyperparameter tuning, no tests.
- `demo.py` and the second `readme.md` are leftover conda-lab stubs.
