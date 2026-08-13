import os

from huggingface_hub import login


token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not token:
    raise RuntimeError(
        "Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN before running hf_login.py."
    )

login(token=token)
