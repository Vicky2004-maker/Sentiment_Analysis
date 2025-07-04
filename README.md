# Sentiment Analysis

This repository contains a simple sentiment classifier built with an XGBoost classifier on the **Amazon US Wireless Reviews** dataset. It demonstrates a complete pipeline from preprocessing and vectorization with TF‑IDF to model training and evaluation. Hyperparameter tuning is performed using `hyperopt`.

## Contents

- `main_core/sentiment_analysis_final.py` – End‑to‑end training and inference script
- `models/` – Pretrained XGBoost model
- `vectorizer/` – Saved `TfidfVectorizer`
- `.idea/` – Development environment settings (can be ignored)

## Getting Started

### Requirements

The script relies on Python 3.9 and the following packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `nltk`
- `pandarallel`
- `joblib`
- `contractions`
- `seaborn`
- `matplotlib`
- `xgboost`
- `hyperopt`

Create a virtual environment (optional) and install the dependencies from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset

Download the Wireless subset of the **Amazon Customer Reviews** dataset from Amazon's open data repository:

```bash
curl -O https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Wireless_v1_00.tsv.gz
gunzip amazon_reviews_us_Wireless_v1_00.tsv.gz
```

Move the resulting `amazon_reviews_us_Wireless_v1_00.tsv` file to a convenient location (e.g. `data/`). Update the `path_large` variable in `main_core/sentiment_analysis_final.py` to point to this path.

### Training the Model

The repository already contains a trained model (`models/xgb_classifier.model`) and vectorizer (`vectorizer/tfidf.vec`). To retrain from scratch, set `fit = True` near the middle of `main_core/sentiment_analysis_final.py` and run the script:

```bash
python main_core/sentiment_analysis_final.py
```

### Inference

Provide your own review text by setting the `text` variable at the bottom of `main_core/sentiment_analysis_final.py`. Keep `fit = False` so the pretrained artifacts are loaded. The predicted sentiment and evaluation metrics will be printed to the console.

### Evaluation

During training, the script reports accuracy, precision, recall, F1 score, Matthew's correlation coefficient and a full classification report. A confusion matrix heatmap is displayed using `seaborn` and `matplotlib`.

## Repository Structure

```
Sentiment_Analysis/
├── main_core/
│   └── sentiment_analysis_final.py
├── models/
│   └── xgb_classifier.model
├── vectorizer/
│   └── tfidf.vec
└── .idea/
    └── *project settings*
```

## Notes

- Preprocessing expands contractions, removes punctuation and digits, tokenizes with NLTK and lemmatizes tokens using `WordNetLemmatizer`.
- `pandarallel` enables parallel preprocessing. Adjust the worker count to suit your machine.
- The dataset is fairly large (150 MB+). Ensure you have sufficient disk space and memory for vectorization.

## License

This project is for educational use only. Refer to the [Amazon reviews dataset license](https://docs.aws.amazon.com/opendata/latest/opendata/amazon-reviews-pds.html) for restrictions on redistribution.
