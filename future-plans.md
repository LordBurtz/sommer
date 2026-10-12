# TODOS:
Sort by priority?
- Include `noincl` datapoints, recheck balancing + maybe add safeguards

# Future plans

## Data cleaning
Currently we are excluding data points acquisited by `noincl` - victor told FK explicitly to include those

## Benchmarking
High level idea
```python
def compare(description: str, predict_func: (sequence: str, precursor_charge: int) -> [float]):
    testsplit = pd.read_parquet("clean_spectra_test_split.parquet")

    loss = 0

    for point in testsplit:
        loss = add(loss, loss_of(predict_func(point)))

    loss_data = loss_inspect(loss) # sort into buckets: length x charge x mods

    log(timestamp(), description, loss_data)

    print_most_recent_and_top5_from_log()
```

