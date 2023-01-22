import pyaudio
import queue

input_device_index = -1
# 配置录音参数
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

LastEndD = 0
import torch
torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
vad_model, funcs = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
        )
detect_speech = funcs[0]

import whisper
import opencc
import numpy as np
cc = opencc.OpenCC('t2s')
model = whisper.load_model("base")
options = whisper.DecodingOptions(language='zh')

def get_audio_text(audio_data):
    audio = whisper.pad_or_trim(audio_data)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    result = whisper.decode(model, mel, options)
    print(cc.convert(result.text))


# 创建队列用于存储录音数据
q = queue.Queue()

def detect_voice_activity(audio):
    speeches = detect_speech(
        audio, vad_model, sampling_rate=16000
    )
    print(speeches,len(audio))

    if len(speeches) == 2:
        if speeches[1]['start'] - speeches[0]['end'] < 8000:
            return [{"start": speeches[0]['start'], "end": speeches[1]['end']}]
    return speeches

def recording_callback(in_data, frame_count, time_info, status):
    # 将录制的数据存入队列
    q.put(in_data)
    return (in_data, pyaudio.paContinue)

def record():
    p = pyaudio.PyAudio()
    # 检测系统扬声器
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if '立体声混音' in dev['name']:
            input_device_index = i
            break
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    # input_device_index = input_device_index,
                    frames_per_buffer=CHUNK,
                    stream_callback=recording_callback)
    # 开始录音
    stream.start_stream()
    alldata = np.array([],np.float32)
    temp = np.array([],np.float32)
    data = b''
    Recording = False
    while True:
        while len(data) < 2*RATE*2:
            data += q.get()
        temp = np.concatenate((temp, np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0), axis=0)
        
        speeches = detect_voice_activity(temp)
        if len(speeches) == 0:
            if Recording:
                Recording = False
                alldata = np.concatenate((alldata,temp), axis=0)
                get_audio_text(alldata)
            alldata = np.array([],np.float32)
            data = b''
            temp = np.array([],np.float32)
            LastEndD = 0
        elif len(speeches) == 1:
            Recording = True
            start = int(speeches[0]['start'])
            end = int(speeches[0]['end'])

            if start + LastEndD < 8000:
                # 这是一句话
                alldata = np.concatenate((alldata,temp[:end]), axis=0)
                temp = temp[end:]
                get_audio_text(alldata)
                data = b''
                LastEndD = len(temp) - end
            else:
                # 这是两句话
                alldata = temp[:end]
                temp = temp[end:]
                get_audio_text(alldata)
                data = b''
                LastEndD = len(temp) - end
        elif len(speeches) == 2:
            Recording = True
            temp = alldata[int(speeches[0]['end']):]
            alldata = alldata[:int(speeches[0]['end'])]
            data = b''
            get_audio_text(alldata)
            alldata = np.array([],np.float32)
            LastEndD = 0

record()