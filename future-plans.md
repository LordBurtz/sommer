# TODOS:
Sort by priority?
- `torch.set_default_tensor_type` to set device globally for a run

# Future plans

## Data cleaning
`noincl` data points are included now (refac/01 step 6), as victor told FK

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

