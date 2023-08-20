import librosa
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch
import os


def recognize(audio_path, model_path):
    sampling_rate = 16000
    y, sr = librosa.load(audio_path, sr=sampling_rate)

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    inputs = feature_extractor(
        y,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    )

    model = AutoModelForAudioClassification.from_pretrained(model_path)

    with torch.no_grad():
        logits = model(**inputs).logits

    # 获取前五个概率最高的类别及其对应的概率
    num_top_predictions = 5
    top_logits, top_indices = torch.topk(logits, num_top_predictions)
    # top_probs = torch.nn.functional.softmax(top_logits, dim=-1)
    top_probs = torch.sigmoid(top_logits)

    # 将索引转换为标签
    id2label = model.config.id2label
    predicted_labels = [id2label[idx.item()] for idx in top_indices[0]]

    # 打印前五个类别及其概率
    for label, prob in zip(predicted_labels, top_probs[0]):
        print(f"Label: {label}, Probability: {prob:.4f}")

    return predicted_labels, top_probs[0]


# if __name__ == "__main__":

# sampling_rate = 16000
# # 获取当前脚本所在的绝对路径
# current_script_path = os.path.abspath(__file__)
# # 获取当前脚本所在的目录
# current_script_directory = os.path.dirname(current_script_path)
# # 设置当前工作目录为脚本所在的目录
# os.chdir(current_script_directory)

# audio_path = "test.wav"
# y, sr = librosa.load(audio_path, sr=sampling_rate)

# model_path = "checkpoint-95"
# feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
# inputs = feature_extractor(
#     y,
#     sampling_rate=sampling_rate,
#     return_tensors="pt",
# )

# model = AutoModelForAudioClassification.from_pretrained(model_path)

# with torch.no_grad():
#     logits = model(**inputs).logits

# # 获取前五个概率最高的类别及其对应的概率
# num_top_predictions = 5
# top_logits, top_indices = torch.topk(logits, num_top_predictions)
# top_probs = torch.nn.functional.softmax(top_logits, dim=-1)

# # 将索引转换为标签
# id2label = model.config.id2label
# predicted_labels = [id2label[idx.item()] for idx in top_indices[0]]

# # 打印前五个类别及其概率
# for label, prob in zip(predicted_labels, top_probs[0]):
#     print(f"Label: {label}, Probability: {prob:.4f}")
