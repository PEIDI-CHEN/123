import torch
import torch.nn as nn
from kafka import KafkaConsumer

# 加载模型
model_path = 'rnn_model.pth'
model = RNN(len(unique_words), hidden_size, output_size)
model.load_state_dict(torch.load(model_path))
model.eval()

# 准备输入数据的函数
def prepare_input_data(input_sequences, word_to_index):
    max_len = max(len(seq) for seq in input_sequences)
    input_tensor = torch.zeros(len(input_sequences), max_len, len(word_to_index))
    for i, seq in enumerate(input_sequences):
        for j, word_index in enumerate(seq):
            input_tensor[i, j, word_index] = 1
    return input_tensor

# Kafka consumer 配置
bootstrap_servers = 'localhost:9092'
topic = 'nas-topic'

# 创建 Kafka consumer
consumer = KafkaConsumer(topic, bootstrap_servers=bootstrap_servers, auto_offset_reset='earliest', enable_auto_commit=True)

# 处理输入数据并进行推理
for message in consumer:
    input_data = message.value.decode('utf-8').strip().split()
    input_sequences = [[word_to_index[word.lower()] for word in input_data if word.lower() in word_to_index]]

    # 准备输入数据
    input_tensor = prepare_input_data(input_sequences, word_to_index)

    # 执行推理
    with torch.no_grad():
        output = model(input_tensor)

    # 处理输出
    predicted_labels = torch.argmax(output, dim=1)
    for label in predicted_labels:
        if label.item() == 1:
            print("Out of order")
        else:
            print("In order")

                                                
