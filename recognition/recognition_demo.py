from recognition import recognize
import os

if __name__ == "__main__":
    # 如果不加这几行，会报错：OSError: [Errno 2] No such file or directory: 'test.wav'

    # 获取当前脚本所在的绝对路径
    current_script_path = os.path.abspath(__file__)
    # 获取当前脚本所在的目录
    current_script_directory = os.path.dirname(current_script_path)
    # 设置当前工作目录为脚本所在的目录
    os.chdir(current_script_directory)

    # 音频最好使用5s，因为训练集全是5s的音频
    audio_path = "test.wav"
    model_path = r"../checkpoint-95"
    predicted_labels, top_probs = recognize(audio_path, model_path)

    # 打印前五个类别及其概率
    print(predicted_labels)
    print(top_probs)
