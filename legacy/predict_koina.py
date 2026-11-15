import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

import common as c

ROOT = Path(__file__).resolve().parent
MODELS = (
    # 'AlphaPeptDeep_ms2_generic',
    # 'Altimeter_2024_intensities',
    # 'ms2pip_HCD2021',
    # 'Prosit_2025_intensity_22PTM',
    'Prosit_2020_intensity_CID',
    )
BATCH_SIZE = 1000
COLLISION_ENERGY = 35.0
FRAGMENTATION_TYPE = 'CID'
INSTRUMENT_TYPE = 'QE'


def predict(batch, model):
    inputs = [
        ('peptide_sequences', 'BYTES', batch['peptide_sequence'].tolist()),
        ('precursor_charges', 'INT32', batch['precursor_charge'].tolist()),
    ]
    if model in ('Prosit_2025_intensity_22PTM', 'AlphaPeptDeep_ms2_generic', 'Altimeter_2024_intensities'):
        inputs.append(('collision_energies', 'FP32', [COLLISION_ENERGY] * len(batch)))
    if model == 'Prosit_2025_intensity_22PTM':
        inputs.append(('fragmentation_types', 'BYTES', [FRAGMENTATION_TYPE] * len(batch)))
    if model == 'AlphaPeptDeep_ms2_generic':
        inputs.append(('instrument_types', 'BYTES', [INSTRUMENT_TYPE] * len(batch)))
    payload = {'inputs': [
        {'name': name, 'datatype': datatype, 'shape': [len(batch), 1], 'data': values}
        for name, datatype, values in inputs
    ]}

    request = Request(
        f'https://koina.wilhelmlab.org/v2/models/{model}/infer',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'},
    )
    with urlopen(request, timeout=120) as response:
        outputs = json.load(response)['outputs']

    result = batch.copy()
    for output in outputs:
        name = 'annotation' if output['name'] == 'annotations' else output['name']
        result[name] = np.asarray(output['data']).reshape(output['shape']).tolist()
    return result


def score(predictions):
    rows = []
    for row in predictions.itertuples():
        fragments = [
            (match.group(1), max(0.0, intensity))
            for annotation, intensity in zip(row.annotation, row.intensities)
            if (match := re.fullmatch(r'([by]\d+)(?:\+1|\^1)?', annotation))
        ]
        if not fragments:
            raise ValueError('No singly charged b/y annotations recognized.')
        predicted = c.ions_to_vector(
            [ion for ion, intensity in fragments],
            [intensity for ion, intensity in fragments], normalize=False,
        )
        target = c.ions_to_vector(row.matched_ions, row.intensities_raw)
        mask = c.valid_slot_mask(row.peptide_sequence)
        rows.append({
            'precursor_charge': row.precursor_charge,
            'peptide_sequence': row.peptide_sequence,
            'peptide_length': c.peptide_length(row.peptide_sequence),
            'spectral_angle': c.spectral_angle(target, predicted, mask),
            'pearson': c.pearson_corr(target, predicted, mask),
        })
    return pd.DataFrame(rows)


def plot_charge2_sa_error_by_length(metrics, model):
    import matplotlib.pyplot as plt

    frame = metrics.loc[metrics['precursor_charge'].eq(2)].copy()
    modification_codes = frame['peptide_sequence'].str.findall(r'UNIMOD:(\d+)').map(
        lambda codes: frozenset(map(int, codes))
    )
    categories = {
        frozenset(): 'Unmodified',
        frozenset({4}): 'Modified:4',
        frozenset({35}): 'Modified:35',
        frozenset({4, 35}): '4&35',
    }
    frame['modification'] = modification_codes.map(categories)
    frame = frame.dropna(subset=['modification'])
    if frame.empty:
        print(f'No charge-2 spectra in the four requested modification groups for {model}.')
        return None

    frame['sa_error'] = 1 - frame['spectral_angle']
    summary = frame.groupby(['peptide_length', 'modification'], observed=True).agg(
        mean_error=('sa_error', 'mean'), sem=('sa_error', 'sem'), n=('sa_error', 'size'),
    ).reset_index()
    colors = {
        'Unmodified': '#0072B2', 'Modified:4': '#D55E00',
        'Modified:35': '#009E73', '4&35': '#CC79A7',
    }
    markers = {'Unmodified': 'o', 'Modified:4': 's', 'Modified:35': '^', '4&35': 'D'}
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for category in categories.values():
        group = summary.loc[summary['modification'].eq(category)]
        if group.empty:
            continue
        axis.errorbar(
            group['peptide_length'], group['mean_error'], yerr=group['sem'].fillna(0),
            label=f'{category} (n={group["n"].sum():,})', color=colors[category],
            marker=markers[category], linewidth=1.6, markersize=5, capsize=2,
        )
    axis.set(
        xlabel='Peptide length (residues)', ylabel='Mean SA error (1 - spectral angle)',
        title=f'{model}: charge 2', ylim=(0, 1),
    )
    axis.legend(title='Modification')
    axis.grid(axis='y', alpha=0.25)
    plt.show()
    return figure, axis


def print_summary(summary, metrics):
    print('\nValidation accuracy (singly charged b/y ions; higher is better):')
    print(metrics.groupby('model').agg(**summary).round(4).to_string(), flush=True)
    print('\nBy precursor charge:')
    print(metrics.groupby(['model', 'precursor_charge']).agg(**summary).round(4).to_string(), flush=True)



def main():
    path = ROOT / 'refac' / 'clean_spectra_val_split.parquet'
    if set(pd.read_parquet(path, columns=['split'])['split']) != {'val'}:
        raise ValueError('Expected validation rows only.')
    spectra = pd.read_parquet(path, columns=[
        'peptide_sequence', 'precursor_charge', 'matched_ions', 'intensities_raw',
    ])
    spectra = spectra[spectra['precursor_charge'].isin([2, 3, 4])]
    print(f'Assumed inputs where required: NCE={COLLISION_ENERGY}, instrument={INSTRUMENT_TYPE}')
    sumaries = []
    for model in MODELS:
        scores = []
        batch_size = min(BATCH_SIZE, 100) if model == 'Altimeter_2024_intensities' else BATCH_SIZE
        for start in range(0, len(spectra), batch_size):
            batch = predict(spectra.iloc[start:start + batch_size], model)
            scores.append(score(batch).assign(model=model))
            print(f'{model}: {min(start + batch_size, len(spectra))}/{len(spectra)}', flush=True)
        metrics = pd.concat(scores, ignore_index=True)
        plot_charge2_sa_error_by_length(metrics, model)
        summary = dict(n=('spectral_angle', 'size'),
                       mean_sa=('spectral_angle', 'mean'),
                       mean_pearson=('pearson', 'mean'))
        sumaries += [(summary, metrics)]

        print_summary(summary, metrics)

    print("==================")
    for summary, metric in sumaries:
        print_summary(summary, metric)



if __name__ == '__main__':
    main()
