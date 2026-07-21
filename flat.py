import json
f = open("/home/yhmin/SNU-AI-Challenge/Ver11/runs/sft32b_v11/train_log.jsonl")
lines = f.readlines()
last = json.loads(lines[-1])
s = last["step"]
e = last["elapsed_s"]
rem = (e/s)*(2000-s)
h = int(rem//3600)
m = int((rem%3600)//60)
print("Step:", s, "/ 2000")
print("ETA:", h, "hours", m, "minutes")
