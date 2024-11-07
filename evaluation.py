import os
import pandas as pd
import time
import torch
from torch.nn import DataParallel
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.float16, device_map="auto"
)

print(torch.cuda.device_count()) 
# if torch.cuda.device_count() > 1:
#     model = torch.nn.DataParallel(model).module

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

text_content = "我所提供的图片是某个网站的快照，我希望你仔细观察快照内部的图像信息，网站结构和文字语义，并对其进行分类，候选类别有['正常图片', ‘涉黄图片’, '涉政图片', '涉赌图片‘， '涉诈图片']，请你从里面选一个类别作为图像标签，对于由于访问失败导致出现的空白网页或者html代码，请你输出'空白'。例如，我提供给你一张涉黄图片，你应该回答：涉黄图片。"
ids = []
results = []
times = []
index = 1
folder_path = 'samples/'
outout_path = 'inference_results.csv'
all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
count = 1

for file_name in all_files:
    image_path = os.path.join(folder_path, file_name)
    start_time = time.time()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": text_content},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    ids.append(file_name)
    results.append(output_text)
    times.append(elapsed_time)
    print('{} / {} processing... '.format(count, len(all_files)) + file_name + ': ' + str(output_text) + ' 运行时间: {:.4f}s'.format(elapsed_time))

    if count % 50 == 0:
        df = pd.DataFrame({
            "ID": ids,
            "Result": results,
            "Time (s)": times
        })
        if not os.path.isfile(outout_path):
            df.to_csv(outout_path, index=False)
        else:
            df.to_csv(outout_path, mode='a', header=False, index=False)
        # with open(outout_path, 'a') as f:
        #     f.write('\n')  # 在文件末尾添加换行符
        print('已写入{}个文件'.format(count))
        ids = []
        results = []
        times = []
    count += 1

df = pd.DataFrame({
    "ID": ids,
    "Result": results,
    "Time (s)": times
})
if not os.path.isfile(outout_path):
    df.to_csv(outout_path, index=False)
else:
    df.to_csv(outout_path, mode='a', header=False, index=False)
with open(outout_path, 'a') as f:
    f.write('\n')  # 在文件末尾添加换行符