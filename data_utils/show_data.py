import pickle 

data1 = pickle.load(open("all_transformed_pdb_string/DrugBank/A0A0E1R3H3.pkl", "rb"))

print(1)
data2 = pickle.load(open("/experiments/data_transformed_pdb/all_transformed_pdb_string_skip_9/DrugBank/A0A0E1R3H3.pkl", "rb"))
import os

folder_path = '/your/folder/path'  # 替换为你的目录路径

file_count = sum(len(files) for _, _, files in os.walk("/experiments/all_transformed_pdb_string"))
print(f"递归统计，'{folder_path}' 下共有 {file_count} 个文件")

folder_path = '/your/folder/path'  # 替换为你的目录路径

file_count = sum(len(files) for _, _, files in os.walk("/hy-tmp/avail_pdb"))
print(f"递归统计，'{folder_path}' 下共有 {file_count} 个文件")
