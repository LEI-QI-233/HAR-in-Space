import json
from bert_score import score

# ground truth
with open("PATH/TO/GT_JSON", "r", encoding="utf-8") as f_gt:
    gt_list = json.load(f_gt)

gt_dict = {item["video_id"]: item["caption"] for item in gt_list}

# prediction
with open("PATH/TO/PRED_JSON", "r", encoding="utf-8") as f_pred:
    pred_dict = json.load(f_pred)

common_ids = set(gt_dict.keys()) & set(pred_dict.keys())
references = [gt_dict[vid] for vid in common_ids]
candidates = [pred_dict[vid] for vid in common_ids]

P, R, F1 = score(candidates, references, lang="en", verbose=True)

print(f"Avg Precision: {P.mean().item():.4f}")
print(f"Avg Recall:    {R.mean().item():.4f}")
print(f"Avg F1 Score:  {F1.mean().item():.4f}")
