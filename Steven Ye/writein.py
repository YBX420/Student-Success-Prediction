import pandas as pd

file_path = 'different_rfe_result.csv'
df = pd.read_csv(file_path)
new_row = pd.DataFrame({
    'N_feature': [1],
})
df = pd.concat([df,new_row], ignore_index=True)
df.to_csv(file_path, index=False)