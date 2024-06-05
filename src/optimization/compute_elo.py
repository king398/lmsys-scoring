import pandas as pd
from itertools import combinations
import json

df = pd.read_csv("/home/mithil/PycharmProjects/lmsys-scoring/data/train_folds.csv")
merged_models = pd.concat([df['model_a'], df['model_b']]).unique()

combinations_set = set()
for combo in combinations(merged_models, 2):
    sorted_combo = tuple(sorted(combo))
    combinations_set.add(sorted_combo)

combinations_list = list(combinations_set)


def find_rows_with_combination(df, model_a, model_b):
    condition = (
            ((df['model_a'] == model_a) & (df['model_b'] == model_b)) |
            ((df['model_a'] == model_b) & (df['model_b'] == model_a))
    )
    return df[condition]


model_combination_weight_dict = {}
for models in combinations_list:
    win_dict = {models[0]: 1, models[1]: 1, "tie": 1}
    rows = find_rows_with_combination(df, models[0], models[1])
    for index, row in rows.iterrows():
        match row['label']:
            case 0:
                model_id = row['model_a']
                win_dict[model_id] += 1
            case 1:
                model_id = row['model_b']
                win_dict[model_id] += 1
            case 2:
                win_dict["tie"] += 1

    weights_array = list(win_dict.values())
    if sum(weights_array) > 3:
        model_combination_weight_dict.update({f"{models[0]} {models[1]}": weights_array})

with open("/home/mithil/PycharmProjects/lmsys-scoring/data/elo_weights.json", "w") as f:
    json.dump(model_combination_weight_dict, f)
