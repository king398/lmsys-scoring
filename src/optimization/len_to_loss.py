import pandas as pd
from transformers import AutoTokenizer

df = pd.read_csv('/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds_llama.csv')
pred_df = pd.read_csv('/home/mithil/PycharmProjects/lmsys-scoring/trial.csv')
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")

df['len'] = df['text'].apply(lambda x: len(tokenizer(x)['input_ids']))
id_df_dict = dict(zip(df['id'], df['len']))
pred_df['len'] = pred_df['id'].map(id_df_dict)
# corr between log loss and length
pred_df['len_bin'] = pd.qcut(pred_df['len'], q=100)

# Calculate average log loss for each bin
binned_avg_loss = pred_df.groupby('len_bin')['log_loss'].mean().reset_index()

# Sort by the bin's left edge for better readability
binned_avg_loss['left_edge'] = binned_avg_loss['len_bin'].apply(lambda x: x.left)
binned_avg_loss = binned_avg_loss.sort_values('left_edge')

# Print the results
print("Average Log Loss for each length bin:")
for _, row in binned_avg_loss.iterrows():
    print(f"Bin {row['len_bin']}: Average Log Loss = {row['log_loss']:.4f}")

# Calculate overall statistics
