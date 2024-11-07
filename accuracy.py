import pandas as pd

df1 = pd.read_csv('inference_results.csv')
df2 = pd.read_csv('annotations_1.csv')
log_messages = []
prompt = "我所提供的图片是某个网站的快照，我希望你仔细观察快照内部的图像信息，网站结构和文字语义，并对其进行分类，候选类别有['正常图片', ‘涉黄图片’, '涉政图片', '涉赌图片‘， '涉诈图片']，请你从里面选一个类别作为图像标签，对于由于访问失败导致出现的空白网页或者html代码，请你输出'空白'。例如，我提供给你一张涉黄图片，你应该回答：涉黄图片。"
info = "使用2张v100，将网站快照压缩到1700*1700分辨率，随机选取50个样本进行测试："
log_messages.append(info)
log_messages.append('prompt:')
log_messages.append(prompt)

dict = {"['空白']": 0, "['正常图片']": 1, "['涉黄图片']": 2, "['涉政图片']": 3, "['涉赌图片']": 4, "['涉诈图片']": 5}
inverse_dict = ["['空白']", "['正常图片']", "['涉黄图片']", "['涉政图片']", "['涉赌图片']", "['涉诈图片']"]
accuracy = 0
for i in range(len(df1)):
    res = df1['Result'][i]
    if dict[res] == int(df2['Annotation'][i]):
        accuracy += 1
    else:
        log_messages.append('识别错误样例: ' + df1['ID'][i])
        log_messages.append('LLM输出: ' + df1['Result'][i] + ' 标注: ' + inverse_dict[df2['Annotation'][i]])
log_messages.append('accuracy: ' + str(accuracy / len(df1)))

with open('info.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_messages))