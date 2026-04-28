import pandas as pd

file_input = 'wine.csv'
df = pd.read_csv(file_input)
df = df.drop(columns=['iter', 'dataset', 'classical_total_s', 'quantum_total_s'])

aggregations = {
    'test_samples': 'first',
    'original_dim': 'first',
    'selected_dim': 'first',
    'train_samples': ['mean', 'std'],
    'classical_accuracy': ['mean', 'std'],
    'classical_train_s': ['mean', 'std'],
    'classical_predict_s': ['mean', 'std'],
    'quantum_accuracy': ['mean', 'std'],
    'quantum_train_s': ['mean', 'std'],
    'quantum_predict_s': ['mean', 'std'],
    'train_peak_python_mib': ['mean', 'std'],
    'predict_peak_python_mib': ['mean', 'std']
}

df_aggregated = df.groupby('batch_size').agg(aggregations)
df_aggregated.columns = [
    f"{col[0]}_{col[1]}" if isinstance(col, tuple) and col[1] != 'first' else col[0] 
    for col in df_aggregated.columns.values
]
df_aggregated.rename(columns=lambda x: x.replace('_first', ''), inplace=True)
df_aggregated.to_csv(f'{file_input[:-4]}_mean.csv')
