import os

exp_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
print(exp_name)
run_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RunModel.py")
print(run_path)
import subprocess

run_model = f"nohup /usr/local/miniconda3/bin/python {run_path} >run_log/{exp_name}.log 2>&1& echo $!"
# shut_down = "nohup /usr/local/bin/python ./shutdown.py > run_log/shutdown.log 2>&1&"


process = subprocess.Popen(run_model, shell=True, stdout=subprocess.PIPE)

pid = process.communicate()[0].strip().decode()
with open("processing_pid.txt", "a", encoding="utf-8") as f:
    f.write(f"{pid}\n")