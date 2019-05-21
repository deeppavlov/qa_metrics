## Count ODQA metrics

# Usage

```bash
pip install -r requirements.txt
python count_metrics.py <PATH_TO_DATASET> -n <TOP_N_DOCS_RANGE> -ak <ANSWER_KEY>
```

# Example
```bash
python count_metrics.py dataset.csv -n 1 5 -ak 'Answer Toloker 1'
```

The evaluation result is written to `metrics.log`
