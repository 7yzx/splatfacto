import subprocess
import os
import shutil
from rich.console import Console
CONSOLE = Console(width=120)


# 定义多个数据目录
base_dir = r"D:\yzx_code\test_data\t-4-8-fastrecon"

# name_id = [dir_id for dir_id in os.listdir(base_dir)]

# name_id = ['1743413530732.823']
# name_id = ['1743413530732.818', '1743413530732.823', '1743413530732.821', '1743413530732.822', '1743413530732.820', '1743413530732.819']
# name_id = ['2025-03-09_052128']
name_id = ['1744081968053.763']

# 遍历数据目录并执行训练
for id in name_id:
    CONSOLE.print("[Green]Running model id {}".format(id))
    # 设置参数
    dataws_dir = os.path.join(base_dir,id)
    model_path = os.path.join(dataws_dir,"outputs","quick_test-d30-i3k-p100w-normal-bil")
    # 构造命令
    cmd_run_nerfstudio = [
        "python", "run_nerfstudio_quick.py",
        "-s", base_dir,
        "--model_id",id,
        "--model_path",model_path,
        "--image_path", r"D:\yzx_code\test_data\t-4-8-fastrecon\{}\data".format(id)
    ]

    # 运行 subprocess
    try:
        subprocess.run(cmd_run_nerfstudio, check=True)
        CONSOLE.print("[green]Successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        exit(1)





