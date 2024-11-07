import os
import random
import shutil
import pandas as pd
from PIL import Image

def select_random_files(source_folder, destination_folder, num_files, ifannotation=True):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder, exist_ok=True)
    all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    selected_files = random.sample(all_files, min(num_files, len(all_files)))
    sorted_files = sorted(selected_files, key=lambda x: int(os.path.splitext(x)[0]))
    annotations = []
    destination_files = []
    count = 0
    for file_name in sorted_files:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        shutil.copy2(source_path, destination_path)
        if ifannotation:
            try:
                print("数据类型['空白': 0, '正常图片': 1, ‘涉黄图片’: 2, '涉政图片': 3, '涉赌图片‘: 4, '涉诈图片': 5]")
                annotation = input("输入"+file_name+"图片的标注: ")
                destination_files.append(destination_path)
                annotations.append(annotation)
                count += 1
            except Exception as e:
                print(f"无法打开图片 {destination_path}: {e}")
            print(destination_path+ ' : ' + annotation)
        else:
            print(destination_path)

    print(f"已复制 {len(selected_files)} 个文件到 {destination_folder}")
    if ifannotation:
        df = pd.DataFrame({
            "Index": destination_files,
            "Annotation": annotations
        })
        df.to_csv("annatations_1.csv", index=False)

# 示例用法
random.seed(412)
source_folder = "processed_img1_1800"  # 替换为你的源文件夹路径
destination_folder = "samples_1"  # 替换为你的目标文件夹路径
num_files_to_select = 50  # 替换为你想随机选择的文件数量

select_random_files(source_folder, destination_folder, num_files_to_select, False)