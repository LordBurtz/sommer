"""Fragment-intensity prediction for CID MS/MS: data, models, training, comparison.

Pipeline (see README):
  refac/01_data_cleansing.ipynb   merge, clean, split
  refac/02_precursor_charge.ipynb impute missing precursor charges -> spectra_with_charge.parquet
  refac/03_train.ipynb            train models via sommer.training
  refac/04_compare.ipynb          compare runs against each other and Koina via sommer.compare
"""
