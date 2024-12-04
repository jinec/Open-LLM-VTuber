import io
import re
import torch
import numpy as np
import soundfile as sf
from funasr import AutoModel
from .asr_interface import ASRInterface


# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need


class VoiceRecognition(ASRInterface):

    def __init__(
        self,
        model_name: str = "iic/SenseVoiceSmall",
        language: str = "auto",
        vad_model: str = "fsmn-vad",
        punc_model=None,
        ncpu: int = None,
        hub: str = None,
        device: str = "cpu",
        disable_update: bool = True,
        sample_rate: int = 16000,
        use_itn: bool = False,
    ) -> None:

        self.model = AutoModel(
            model=model_name,
            vad_model=vad_model,
            ncpu=ncpu,
            hub=hub,
            device=device,
            disable_update=disable_update,
            punc_model=punc_model,
            # spk_model="cam++",
        )
        self.SAMPLE_RATE = sample_rate
        self.use_itn = use_itn
        self.language = language

        self.asr_with_vad = None
        self.hotwords = self.__load_hotwords()
        self.generalization_dict = self.__load_GeneralizedWords()

    # Implemented in asr_interface.py
    # def transcribe_with_local_vad(self) -> str:

    def __load_hotwords(self,):
        with open("asr/hotwords.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        return " ".join(lines)
    
    def __load_GeneralizedWords(self,):
        generalizedWords_dictory = {}
        with open("asr/generalizedWords.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line_words = line.strip().split(" ")
                if len(line_words)==2:
                    generalizedWords_dictory[line_words[0]] = line_words[1]
        return generalizedWords_dictory

    def transcribe_np(self, audio: np.ndarray) -> str:

        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        res = self.model.generate(
            input=audio_tensor,
            batch_size_s=300,
            use_itn=self.use_itn,
            language=self.language,
            hotword=self.hotwords
        )

        full_text = res[0]["text"]

        # SenseVoiceSmall may spits out some tags
        # like this: '<|zh|><|NEUTRAL|><|Speech|><|woitn|>欢迎大家来体验达摩院推出的语音识别模型'
        # we should remove those tags from the result

        # remove tags
        full_text = re.sub(r"<\|.*?\|>", "", full_text)
        # the tags can also look like '< | en | > < | EMO _ UNKNOWN | > < | S pe ech | > < | wo itn | > ', so...
        full_text = re.sub(r"< \|.*?\| >", "", full_text)

        # 泛化词典处理        
        matches = []# 收集所有的匹配位置
        for key in sorted(self.generalization_dict, key=len, reverse=True):  # 按长度降序排列
            start = 0
            while True:
                start = full_text.find(key, start)
                if start == -1:
                    break
                matches.append((start, start + len(key), key))
                start += len(key)

        # 对匹配项按结束位置排序，以便从后往前替换
        matches.sort(key=lambda x: x[1], reverse=True)

        # 执行替换
        for start, end, key in matches:
            full_text = full_text[:start] + self.generalization_dict[key] + full_text[end:]

        return full_text.strip()

    def _numpy_to_wav_in_memory(self, numpy_array: np.ndarray, sample_rate):

        memory_file = io.BytesIO()
        sf.write(memory_file, numpy_array, sample_rate, format="WAV")
        memory_file.seek(0)

        return memory_file
