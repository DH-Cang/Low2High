from huggingface_hub import HfApi
api = HfApi()
api.create_repo(repo_id="cdh-sd-normal", private=True)
api.upload_folder(
    folder_path="/path/to/local/folder",
    repo_id="cdh573885/cdh-sd-normal",
    ignore_patterns="**/logs/*.txt", # List[str]
    )