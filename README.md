# Machine Learning Project
Machine learning project performed during the neue fische Data Science Bootcamp. The time window was three workdays.
- The notebook containing the EDA and the error analysis can be found [here](EDA-and-error-analysis.ipynb).
- The modelling takes place in [this notebook](modelingXGB.ipynb).
- Scripts to load and preprocess the data are collected [here](prepare_flight_data.py).
- Scripts and custom transformer to create features can be found [here](feature_engineering.py).
- 
## Dataset
The project is using [Flight data from zindi challenge](https://zindi.africa/competitions/ai-tunisia-hack-5-predictive-analytics-challenge-2/data) and [geographical airports data](https://pypi.org/project/airportsdata/). Download the data to the data folder.

## Requirements
- pyenv
- python==3.9.4
## Setup
Use the `makefile`to install the requirements.
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
