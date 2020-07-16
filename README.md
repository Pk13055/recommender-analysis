# recommender-analysis

Exploratory app on building a production recommender pipeline

## Setup

```bash
git clone https://github.com/Pk13055/recommender-analysis.git
cd recommender-analysis

mkdir data && cd data
wget http://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip
mv ml-25m/* .
rm -rf ml-25*
```


## Installation

```bash
pip install -r requirements.txt
```
## Running

```bash
streamlit run app.py
```

## Development

```bash
pip install pip-tools
pip-compile requirements.in > requirements.txt
pip-sync # OR pip install -r requirements.txt

streamlit run app.py
```

