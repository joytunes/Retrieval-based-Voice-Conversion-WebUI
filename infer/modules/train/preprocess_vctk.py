import multiprocessing
import os
import sys

sys.path.append("/home/ubuntu/rvc")
import vctk

from scipy import signal

now_dir = os.getcwd()
sys.path.append(now_dir)
print(sys.argv)
inp_root = sys.argv[1]
sr = int(sys.argv[2])
n_p = int(sys.argv[3])
exp_dir = sys.argv[4]
noparallel = sys.argv[5] == "True"
per = float(sys.argv[6])
import multiprocessing
import os
import traceback

import librosa
import numpy as np
from scipy.io import wavfile

from infer.lib.audio import load_audio
from infer.lib.slicer2 import Slicer

mutex = multiprocessing.Lock()
f = open("%s/preprocess.log" % exp_dir, "a+")


def println(strr):
    mutex.acquire()
    print(strr)
    f.write("%s\n" % strr)
    f.flush()
    mutex.release()


class PreProcess:
    def __init__(self, sr, exp_dir, per=3.7):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.per = per
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.exp_dir = exp_dir
        self.gt_wavs_dir = "%s/0_gt_wavs" % exp_dir
        self.wavs16k_dir = "%s/1_16k_wavs" % exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def norm_write(self, tmp_audio, name: str, idx1: int):
        # print(f"NORM_WRITE {name} {idx1}")
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            print("%s-%s-%s-filtered" % (name, idx1, tmp_max))
            return
        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
            1 - self.alpha
        ) * tmp_audio
        output_name = f"{name}_{idx1:03d}"
        wav_path = "%s/%s.wav" % (self.gt_wavs_dir, output_name)
        assert not os.path.exists(wav_path), f"File already exists: {wav_path}"
        wavfile.write(
            wav_path,
            self.sr,
            tmp_audio.astype(np.float32),
        )
        tmp_audio = librosa.resample(
            tmp_audio, orig_sr=self.sr, target_sr=16000
        )  # , res_type="soxr_vhq"
        wav16k_path = "%s/%s.wav" % (self.wavs16k_dir, output_name)
        assert not os.path.exists(wav16k_path), f"File already exists: {wav16k_path}"
        wavfile.write(
            wav16k_path,
            16000,
            tmp_audio.astype(np.float32),
        )

    def pipeline(self, path, name):
        # print(f"PIPLINE {path} {name}")
        try:
            audio = load_audio(path, self.sr)
            # zero phased digital filter cause pre-ringing noise...
            # audio = signal.filtfilt(self.bh, self.ah, audio)
            # print(f"Original audio shape: {audio.shape}")
            audio = signal.lfilter(self.bh, self.ah, audio)
            # print(f"Filtered audio shape: {audio.shape}")

            idx1 = 0
            for audio in self.slicer.slice(audio):
                # print(f"== slice audio shape: {audio.shape}")
                i = 0
                while 1:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if len(audio[start:]) > self.tail * self.sr:
                        tmp_audio = audio[start : start + int(self.per * self.sr)]
                        # print(f"{start=}, end={start + int(self.per * self.sr)}, {idx1=}")
                        self.norm_write(tmp_audio, name, idx1)
                        idx1 += 1
                    else:
                        # print(f"{start=}, {idx1=}")
                        tmp_audio = audio[start:]
                        self.norm_write(tmp_audio, name, idx1)
                        idx1 += 1
                        break
            println("%s->Suc." % path)
        except:
            println("%s->%s" % (path, traceback.format_exc()))
            raise

    def pipeline_mp(self, infos):
        for row in infos.itertuples():
            row_name = f"{row.speaker_id}_{row.text_id}"
            self.pipeline(row.audio, row_name)

    def pipeline_mp_inp_dir(self, dset, n_p):
        try:
            # infos = [
            #     ("%s/%s" % (inp_root, name), idx)
            #     for idx, name in enumerate(sorted(list(os.listdir(inp_root))))
            # ]
            if noparallel:
                print("Running serially")
                for i in range(n_p):
                    self.pipeline_mp(dset.iloc[i::n_p])
            else:
                print(f"Running {n_p} processes")
                ps = []
                for i in range(n_p):
                    p = multiprocessing.Process(
                        target=self.pipeline_mp, args=(dset.iloc[i::n_p],)
                    )
                    ps.append(p)
                    p.start()
                for i in range(n_p):
                    ps[i].join()
                    if ps[i].exitcode != 0:
                        raise Exception(f"Process {i} failed with exit code {ps[i].exitcode}")
        except:
            println("Fail. %s" % traceback.format_exc())
            raise


def preprocess_trainset(inp_root, sr, n_p, exp_dir, per):
    pp = PreProcess(sr, exp_dir, per)
    println("start preprocess")
    println(sys.argv)
    println("Loading vctk dataset...")
    vctk_dset = vctk.get_dataset(inp_root, mic_ids=(2,))
    # vctk_dset = vctk_dset.iloc[7:8]
    println("Dataset loaded")
    pp.pipeline_mp_inp_dir(vctk_dset, n_p)
    println("end preprocess")


if __name__ == "__main__":
    preprocess_trainset(inp_root, sr, n_p, exp_dir, per)
