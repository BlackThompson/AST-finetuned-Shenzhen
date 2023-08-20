# _*_ coding : utf-8 _*_
# @Time : 2023/8/7 12:59
# @Author : Black
# @File : video2wav
# @Project : MultiSoundRecogonition

from moviepy.editor import VideoFileClip
import librosa
import math
import os
import soundfile as sf


def convert_to_wav(input_folder, output_folder):
    """
    将输入文件夹中的音频文件转换为WAV格式，并保存到输出文件夹中
    :param input_folder:
    :param output_folder:
    :return:
    """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的音频文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            continue  # 如果已经是WAV文件，跳过

        input_path = os.path.join(input_folder, filename)
        print(input_path)
        output_name = os.path.splitext(filename)[0] + ".wav"
        output_path = os.path.join(output_folder, output_name)

        try:
            # 使用soundfile进行格式转换
            audio, samplerate = sf.read(input_path)
            sf.write(output_path, audio, samplerate, format="WAV")
            print(f"已转换 {filename} 到 {output_name}")
        except Exception as e:
            print(f"转换 {filename} 时出现错误：{e}")


def split_audio(input_path, output_folder, segment_duration=5):
    """
    将音频文件切割为指定时长的片段，并保存到输出文件夹中
    :param input_path:
    :param output_folder:
    :param segment_duration:
    :return:
    """
    # 读取原始音频
    audio_data, sample_rate = librosa.load(input_path, sr=None)

    # 计算片段数量
    # segment_duration = 10  # 每个片段的时长（秒）
    total_duration = len(audio_data) / sample_rate
    num_segments = math.floor(total_duration / segment_duration)

    if num_segments == 0:
        print("音频时长不足5秒，无需处理")
        return

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 切割并保存音频片段
    for i in range(num_segments):
        start = int(i * segment_duration * sample_rate)
        end = int((i + 1) * segment_duration * sample_rate)
        segment = audio_data[start:end]

        output_name = (
            f"{os.path.splitext(os.path.basename(input_path))[0]}_{i + 1:02d}.wav"
        )
        output_path = os.path.join(output_folder, output_name)

        sf.write(output_path, segment, sample_rate)

    print(f"已切割并保存 {num_segments} 个音频片段到 {output_folder}")


def change_sample_rate(audio_path, output_sample_rate):
    """
    修改音频的采样率
    :param audio_path:
    :param output_sample_rate:
    :return:
    """
    y, _ = librosa.load(audio_path, sr=output_sample_rate)
    sf.write(audio_path, y, output_sample_rate)


def extract_audio_from_video(video_path, output_folder, output_sample_rate=16000):
    """
    从视频中提取音频
    :param video_path:
    :param output_folder:
    :param output_sample_rate:
    :return:
    """
    # 获取视频文件名（不带扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # 使用MoviePy库加载视频文件
    video_clip = VideoFileClip(video_path)
    # 提取音频
    audio_clip = video_clip.audio
    # 构建输出文件路径
    output_path = os.path.join(output_folder, f"{video_name}.wav")
    # 保存音频为wav文件
    audio_clip.write_audiofile(output_path, codec="pcm_s16le")
    # 修改采样率
    change_sample_rate(output_path, output_sample_rate)
    print(f"音频已提取并保存为 {output_path}")
