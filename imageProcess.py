import os
from math import sqrt
from PIL import Image

def compress_images(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            # 构造完整的文件路径
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 打开图片并压缩
            with Image.open(input_path) as img:
                width, height = img.size
                times = 1800 / sqrt(width * height)
                if times < 1:
                    target_size = (int(width * times), int(height * times))
                    img = img.resize(target_size, Image.LANCZOS)
                img.save(output_path)
                print(f"Compressed and saved: {output_path}")

# 使用示例
input_folder = 'images'  # 输入文件夹路径
output_folder = 'processed_images'  # 输出文件夹路径
compress_images(input_folder, output_folder)
