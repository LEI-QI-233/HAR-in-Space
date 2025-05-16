import json
from pathlib import Path
from collections import defaultdict
from bert_score import score

# ---------------- 1. 路径 ----------------
gt_json   = Path(r"C:\Users\charl\Downloads\vpa\vpa\caption_vqa_gt.json")
pred_json = Path(r"C:\Users\charl\Downloads\vpa\vpa\vqa_Qwen25_predictions.json")

# ---------------- 2. 读取并转成  (video_id, Q)  →  A  的字典 ----------------
def build_qa_dict(fp: Path, is_gt: bool):
    """把原始 list ➜ {(vid, question): answer}，若缺 answer 设为空串"""
    qa_dict = {}
    with fp.open(encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        vid = item["video_id"]
        # ground-truth 里还带 caption，但我们只关心 qa_pairs
        for pair in item["qa_pairs"]:
            q = pair["Q"].strip()
            a = pair.get("A", "").strip()
            qa_dict[(vid, q)] = a
    return qa_dict

gt_dict   = build_qa_dict(gt_json,  is_gt=True)
pred_dict = build_qa_dict(pred_json, is_gt=False)

# ---------------- 3. 对齐键值，只评交集 ----------------
common_keys = gt_dict.keys() & pred_dict.keys()       # set 交集
missing_pred = len(gt_dict) - len(common_keys)
missing_gt   = len(pred_dict) - len(common_keys)

if missing_pred:
    print(f"[Warn] {missing_pred} QA pairs 没有预测答案，将被忽略。")
if missing_gt:
    print(f"[Warn] {missing_gt} 预测答案在 GT 中找不到对应问句，也将被忽略。")

references = [gt_dict[key]   for key in common_keys]
candidates = [pred_dict[key] for key in common_keys]

print(f"Evaluate {len(common_keys)} QA pairs")

# ---------------- 4. BERTScore ----------------
P, R, F1 = score(candidates, references, lang="en", verbose=True)

print("\n==========  BERTScore (Answers)  ==========")
print(f"Avg Precision: {P.mean():.4f}")
print(f"Avg Recall:    {R.mean():.4f}")
print(f"Avg F1 Score:  {F1.mean():.4f}")
