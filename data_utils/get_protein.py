import os.path
import time
import pandas as pd 
import requests
from concurrent.futures import ThreadPoolExecutor
MAX_REPEAT = 5
SLEEP = 2


# 爬取网页数据的函数
def get_pdb_id(uniprot_id):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/91.0.4472.124 Safari/537.36'
    }
    url = f'https://rest.uniprot.org/uniprotkb/{uniprot_id}'
    repeat = 0
    while True:
        try:
            # 构造请求标头
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # 检查请求是否成功
            if response.status_code == 200:
                break
        except requests.exceptions.RequestException as e:
            repeat += 1
            time.sleep(SLEEP)
            if repeat >= MAX_REPEAT:
                print(f"{uniprot_id}uniprot_ip被封")
                return {"error": 1}
    # 解析网页内容
    if response.status_code == 200:
        database_name2index = dict()
        if "uniProtKBCrossReferences" not in response.json():
            return {"error": 0}      # 数据已删除
        database_list = response.json()["uniProtKBCrossReferences"]

        for i, database in enumerate(database_list):
            name = database["database"]
            if name not in database_name2index:
                database_name2index[name] = [i]
            else:
                database_name2index[name].append(i)

        if "PDB" in database_name2index:
            index = database_name2index["PDB"]
            best_resolution = {"Resolution": 10000, "length": 0, "id": ""}
            best_length = {"Resolution": 10000, "length": 0, "id": ""}
            for i in index:
                database = database_list[i]
                assert database["database"] == 'PDB'
                properties = database["properties"]
                current_pro = {"Resolution": 10000, "length": 0, "id": database["id"]}
                for p in properties:
                    if p["key"] == "Resolution":
                        try:
                            current_pro["Resolution"] = float(p["value"].split(" ")[0])
                        except:
                            current_pro["Resolution"] = 10000
                    elif p["key"] == "Chains":
                        [begin, end] = p["value"].split("=")[-1].split("-")
                        if begin != "" and end != "":
                            current_pro["length"] = int(end) - int(begin)
                if current_pro["Resolution"] < best_resolution["Resolution"]:
                    best_resolution.update(current_pro)
                if current_pro["length"] > best_length["length"]:
                    best_length.update(current_pro)
            if best_resolution["length"] >= best_length["length"] - 50 and best_resolution["id"] != '':
                if best_resolution["length"] > response.json()["sequence"]["length"]*0.5:
                    return {"PDB": best_resolution["id"]}
            else:
                if best_resolution["length"] > response.json()["sequence"]["length"]*0.5:
                    return {"PDB": best_length["id"]}
            if "AlphaFoldDB" in database_name2index:
                index = database_name2index["AlphaFoldDB"][0]
                assert database_list[index]["database"] == 'AlphaFoldDB'
                return {"AlphaFoldDB": database_list[index]["id"]}
            else:
                print(f"{uniprot_id}结构数据太短")
                return {"error": 3}      # 现有结构太短

        elif "AlphaFoldDB" in database_name2index:
            index = database_name2index["AlphaFoldDB"][0]
            assert database_list[index]["database"] == 'AlphaFoldDB'
            return {"AlphaFoldDB": database_list[index]["id"]}
        else:
            print(f"{uniprot_id}无三维结构")
            return {"error": 2}  # 无三维结构


def download_from_pdb(uniprot_id, pdb_id, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/91.0.4472.124 Safari/537.36'
    }
    url_1 = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    url_2 = f'https://files.rcsb.org/download/{pdb_id}.cif'
    res_1 = 0
    repeat = 0
    while True:
        try:
            response = requests.get(url_1, headers=headers)
            response.raise_for_status()  # 检查请求是否成功
            if response.status_code == 200:
                res_1 += 1
                break
        except requests.exceptions.RequestException as e:
            repeat += 1
            time.sleep(SLEEP)
            if repeat >= MAX_REPEAT:
                break
    repeat = 0
    if not res_1:
        while True:
            try:
                response = requests.get(url_2, headers=headers)
                response.raise_for_status()  # 检查请求是否成功
                if response.status_code == 200:
                    break
            except requests.exceptions.RequestException as e:
                repeat += 1
                time.sleep(SLEEP)
                if repeat >= MAX_REPEAT:
                    print(f"pdb_ip被封了")
                    return {"error": 1}

    if response.status_code == 200:
        pdb = response.content.replace(b"\r\n", b"\n")
        if res_1:
            with open(os.path.join(save_dir, f"{uniprot_id}.pdb"), "wb") as f:
                f.write(pdb)
        else:
            with open(os.path.join(save_dir, f"{uniprot_id}.cif"), "wb") as f:
                f.write(pdb)
        return {}
    else:
        print(f"无{pdb_id}pdb文件")
        return {"error": 2}


def download_from_alphafold(uniprot_id, pdb_id, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/91.0.4472.124 Safari/537.36'
    }
    url = f'https://alphafold.ebi.ac.uk/files/AF-{pdb_id}-F1-model_v4.pdb'
    repeat = 0
    while True:
        try:
            response = requests.get(url,headers, verify=False)
            response.raise_for_status()  # 检查请求是否成功
            if response.status_code == 200:
                break
        except requests.exceptions.RequestException as e:
            repeat += 1
            time.sleep(SLEEP)
            if repeat >= MAX_REPEAT:
                print(f"Connection to AlphaFold failed. You may need to upgrade OpenSSL to 3.x.x. or change the IP address")
                return {"error": 1}
    if response.status_code == 200:
        pdb = response.content.replace(b"\r\n", b"\n")
        with open(os.path.join(save_dir, f"{uniprot_id}.pdb"), "wb") as f:
            f.write(pdb)
        return {}
    else:
        print(f"无{pdb_id}pdb文件")
        return {"error": 2}


def download_protein(uniprot_id, save_dir, log_dir):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    id_res = get_pdb_id(uniprot_id)
    if "error" in id_res:
        if id_res["error"] == 1:
            with open(os.path.join(log_dir, "网页访问失败.txt"), "a") as f:
                f.write(f"{uniprot_id}\n")
        elif id_res["error"] == 0:
            with open(os.path.join(log_dir, "实体被删除.txt"), "a") as f:
                f.write(f"{uniprot_id}\n")
        elif id_res["error"] == 2:
            with open(os.path.join(log_dir, "无三维结构.txt"), "a") as f:
                f.write(f"{uniprot_id}\n")
        elif id_res["error"] == 3:
            with open(os.path.join(log_dir, "结构太短.txt"), "a") as f:
                f.write(f"{uniprot_id}\n")

    elif "PDB" in id_res:
        down_res = download_from_pdb(uniprot_id, id_res["PDB"], save_dir)
        if "error" in down_res:
            if down_res["error"] == 1:
                with open(os.path.join(log_dir, "网页访问失败.txt"), "a") as f:
                    f.write(f"{uniprot_id}\n")
            else:
                with open(os.path.join(log_dir, "PDB文件下载失败.txt"), "a") as f:
                    f.write(f"{uniprot_id}\n")

    elif "AlphaFoldDB" in id_res:
        down_res = download_from_alphafold(uniprot_id, id_res["AlphaFoldDB"], save_dir)
        if "error" in down_res:
            if down_res["error"] == 1:
                with open(os.path.join(log_dir, "网页访问失败.txt"), "a") as f:
                    f.write(f"{uniprot_id}\n")
            else:
                with open(os.path.join(log_dir, "AlphaFoldDB文件下载失败.txt"), "a") as f:
                    f.write(f"{uniprot_id}\n")


def collect_pdb(dataset_path, save_path, log_path, num_threads):
     for dataset_file in os.listdir(dataset_path):
            dataset = dataset_file[:-4]
            save_path = os.path.join(save_path, dataset)
            print(f"collecting {dataset}...")
            df = pd.read_csv(os.path.join("./InitialDataSets", dataset_file), header=None, names = ["drug_id", "protein_id", "smiles", "seq", "inter"], sep=" ")
            df = df.drop_duplicates(subset="protein_id")
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                para = []
                for p in df["protein_id"]:
                    para.append((p, save_path, log_path))
                executor.map(download_protein, df["protein_id"], [save_path]*len(df), [log_path]*len(df))

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    if 1:
        download_protein("A3FMN7", "temp", "temp")  # 收集指定蛋白质的结构
    if 0:  # 收集所有数据集中的蛋白质结构
        for dataset in os.listdir("./DataSets"):
            if dataset != "Enzyme":
            # if dataset == "DrugBank" or dataset == "Davis":
                continue
            df = pd.read_csv(os.path.join("./DataSets", dataset, "protein.csv"))
            save_dir = f"./handle_data/{dataset}"
            log_dir = f"./log/{dataset}"
            with ThreadPoolExecutor(max_workers=64) as executor:
                # 提交 5 个任务到线程池中
                # for protein in tqdm(df["protein_id"], total=len(df["protein_id"])):
                para = []
                for p in df["protein_id"]:
                    para.append((p, save_dir, log_dir))
                executor.map(download_protein, df["protein_id"], [save_dir]*len(df), [log_dir]*len(df))
                    # download_protein(protein, f"./handle_data/{dataset}", f"./log/{dataset}")
                    # time.sleep(2)
    if 1:
        for dataset in os.listdir("./log"):
            # if dataset != "Enzyme":
            #     # if dataset == "DrugBank" or dataset == "Davis":
            #     continue
            if os.path.exists(os.path.join("./log", dataset, "网页访问失败.txt")):
                with open(os.path.join("./log", dataset, "网页访问失败.txt"), "r") as f:
                    proteins = f.read().splitlines()
                f = open(os.path.join("./log", dataset, "网页访问失败.txt"), "w")
                f.close()
                save_dir = f"./handle_data/{dataset}"
                log_dir = f"./log/{dataset}"
                with ThreadPoolExecutor(max_workers=64) as executor:
                    # 提交 5 个任务到线程池中
                    # for protein in tqdm(df["protein_id"], total=len(df["protein_id"])):
                    executor.map(download_protein, proteins, [save_dir] * len(proteins), [log_dir] * len(proteins))
                    # download_protein(protein, f"./handle_data/{dataset}", f"./log/{dataset}")
                    # time.sleep(2)





