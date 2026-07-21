import json
try:
    with open("runs/sft32b_v11/train_log.jsonl") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    if lines:
        s = lines[-1]["step"]
        e = lines[-1]["elapsed_s"]
        rem = (e/s)*(2000-s)
        print(f"현재: {s}/2000 스텝, 남은시간: {int(rem//3600)}시간 {int((rem%3600)//60)}분")
except Exception as e:
    print(e)
