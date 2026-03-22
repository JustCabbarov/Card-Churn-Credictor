# 💳 Card Activity Churn Predictor

Predicting card disengagement using transaction behavioral patterns.

## About
A machine learning project that predicts whether a bank card customer 
is likely to disengage based on their transaction behavior.

Built as part of my journey into AI/ML development.

## Tech Stack
- Python
- scikit-learn (Random Forest)
- XGBoost
- Streamlit
- pandas, numpy

## Features
| Feature | Description |
|---|---|
| Card age | How long the customer has had the card |
| Avg monthly transactions | Average transactions per month |
| Avg transaction amount | Average transaction value in AZN |
| Months since last transaction | Recency of card usage |
| Transaction trend | -1 (declining) to +1 (growing) |
| Unique merchant count | Diversity of card usage |
| International usage | Whether card is used abroad |
| Complaint count | Number of complaints filed |

## Results
| Model | Precision | Recall | F1 |
|---|---|---|---|
| Random Forest | 0.96 | 0.95 | 0.95 |
| XGBoost | 0.97 | 0.99 | 0.98 |

## Setup
```bash
pip install -r requirements.txt
python Data/genereate_data.py
python train.py
streamlit run app.py
```

## Note
All data is synthetic — no real customer data is used.
