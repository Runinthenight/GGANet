import os
import shutil
from zipfile import ZipFile

def zip_exp_folders(zip_dir, output_zip="exp_folders.zip"):
    current_dir = zip_dir
    exp_folders = [f for f in os.listdir(current_dir) if os.path.isdir(f)]
    if not exp_folders:
        print("No folders starting with 'exp' found.")
        return

    with ZipFile(output_zip, 'w') as zipf:
        for folder in exp_folders:
            for root, _, files in os.walk(folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    # 将文件添加到 ZIP 中，保持目录结构
                    zipf.write(file_path, os.path.relpath(file_path, current_dir))
        for file in os.listdir(current_dir):
            file_path = os.path.join(current_dir, file)
            if os.path.isfile(file_path):
                zipf.write(file_path, os.path.relpath(file_path, current_dir))
        zipf.write("./shutdown.py", os.path.relpath("./shutdown.py", current_dir))
        zipf.write("./auto_shutdown.py", os.path.relpath("./auto_shutdown.py", current_dir))
    print(f"Successfully zipped {len(exp_folders)} folders into {output_zip}")

if __name__ == "__main__":
    zip_exp_folders("/experiments/", "experiments_20250318.zip")
