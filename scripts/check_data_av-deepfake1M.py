from huggingface_hub import hf_hub_download
import os

# Xem cấu trúc thư mục trước
from huggingface_hub import list_repo_files

files = list(list_repo_files(
    "ControlNet/AV-Deepfake1M", 
    repo_type="dataset"
))
print(files[:50])