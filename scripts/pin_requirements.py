"""Pin the key libraries in requirements.txt to the versions the model was TRAINED with.
Run before deploying (Streamlit Cloud / Docker) so the pickled model loads in the same environment:
    python scripts/pin_requirements.py
"""
import json
import re

env = json.load(open("models/meta.json"))["env"]
lines = []
for line in open("requirements.txt").read().splitlines():
    name = re.split(r"[<>=!~ ]", line.strip(), maxsplit=1)[0]
    lines.append(f"{name}=={env[name]}" if name in env and env[name] != "n/a" else line)
open("requirements.txt", "w").write("\n".join(lines) + "\n")
print("pinned:", {k: v for k, v in env.items() if k in "".join(lines)})
