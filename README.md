## Count ODQA metrics

# Available metrics

* ranker em recall
* squad f1 v1
* squad em v1

# Usage

```bash
pip install -r requirements.txt
python count_metrics.py <PATH_TO_DATASET> <DATASET_NAME> -n <TOP_N_DOCS_RANGE> -ak <ANSWER_KEY>
```

# Available datasets

* sddata_pull
* ru_squad

# Example
```bash
python count_metrics.py dataset.csv sddata_pull -n 1 5 -ak 'Answer Toloker 1'
```

The evaluation result is written to `metrics.log`
