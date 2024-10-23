import os
import sys
import traceback
import logging

logger = logging.getLogger(__name__)

from functools import lru_cache
from time import time as ttime

import faiss
import librosa
import numpy as np
import parselmouth
import pyworld
import torch
import torch.nn.functional as F
import torchcrepe

from scipy import signal

from pyloudnorm.iirfilter import IIRfilter

from infer.lib.audio import extract_features_simple, extract_features_simple_segment
from infer.lib.train.mel_processing import spectrogram_torch
from infer.lib.train.utils import load_wav_to_torch

now_dir = os.getcwd()
sys.path.append(now_dir)

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

input_audio_path2wav = {}


@lru_cache
def cache_harvest_f0(input_audio_path, fs, f0max, f0min, frame_period):
    audio = input_audio_path2wav[input_audio_path]
    f0, t = pyworld.harvest(
        audio,
        fs=fs,
        f0_ceil=f0max,
        f0_floor=f0min,
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(audio, f0, t, fs)
    return f0


def k_weighting_filter(data, rate):
    high_shelf = IIRfilter(4.0, 1 / np.sqrt(2), 1500.0, rate, "high_shelf")
    high_pass = IIRfilter(0.0, 0.5, 38.0, rate, "high_pass")
    if len(data.shape) == 1:
        data = high_shelf.apply_filter(data)
        data = high_pass.apply_filter(data)
    else:
        data = data.copy()
        for ch in len(data.shape[0]):
            data[ch] = high_shelf.apply_filter(data[ch])
            data[ch] = high_pass.apply_filter(data[ch])
    return data


def change_rms(data1, sr1, data2, sr2, rate):  # 1是输入音频，2是输出音频,rate是2的占比
    # print(data1.max(),data2.max())
    data1_filtered = k_weighting_filter(data1, sr1)
    data2_filtered = k_weighting_filter(data2, sr2)

    rms1 = librosa.feature.rms(
        y=data1_filtered, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # 每半秒一个点
    rms2 = librosa.feature.rms(
        y=data2_filtered, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2
    )
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (
        torch.pow(rms1, torch.tensor(1 - rate))
        * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()
    return data2


class Pipeline(object):
    def __init__(self, tgt_sr, config, x_query=None):
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half,
        )
        self.sr = 16000  # hubert输入采样率
        self.window = 160  # 每帧点数
        self.t_pad = self.sr * self.x_pad  # 每条前后pad时间
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query  # 查询切点前后查询时间
        self.t_center = self.sr * self.x_center  # 查询切点位置
        self.t_max = self.sr * self.x_max  # 免查询时长阈值
        self.device = config.device

    def get_f0(
        self,
        input_audio_path,
        x,
        p_len,
        f0_up_key,
        f0_method,
        filter_radius,
        f0_invert_axis=None,
        inp_f0=None,
        f0_npy_path="",
    ):
        global input_audio_path2wav
        time_step = self.window / self.sr * 1000
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        if f0_npy_path != "":
            f0 = np.load(f0_npy_path)
        elif f0_method == "pm":
            f0 = (
                parselmouth.Sound(x, self.sr)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    pitch_ceiling=f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
        elif f0_method == "harvest":
            input_audio_path2wav[input_audio_path] = x.astype(np.double)
            f0 = cache_harvest_f0(input_audio_path, self.sr, f0_max, f0_min, 10)
            if filter_radius > 2:
                f0 = signal.medfilt(f0, 3)
        elif f0_method == "crepe":
            model = "full"
            # Pick a batch size that doesn't cause memory errors on your gpu
            batch_size = 512
            # Compute pitch using first gpu
            audio = torch.tensor(np.copy(x))[None].float()
            f0, pd = torchcrepe.predict(
                audio,
                self.sr,
                self.window,
                f0_min,
                f0_max,
                model,
                batch_size=batch_size,
                device=self.device,
                return_periodicity=True,
            )
            pd = torchcrepe.filter.median(pd, 3)
            f0 = torchcrepe.filter.mean(f0, 3)
            f0[pd < 0.1] = 0
            f0 = f0[0].cpu().numpy()
        elif f0_method.startswith("rmvpe"):
            if not hasattr(self, "model_rmvpe"):
                from infer.lib.rmvpe import RMVPE

                logger.info(
                    "Loading rmvpe model,%s" % "%s/rmvpe.pt" % os.environ["rmvpe_root"]
                )
                self.model_rmvpe = RMVPE(
                    "%s/rmvpe.pt" % os.environ["rmvpe_root"],
                    is_half=self.is_half,
                    device=self.device,
                    # use_jit=self.config.use_jit,
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
            if f0_method == "rmvpe_alt":
                model = "full"
                # Pick a batch size that doesn't cause memory errors on your gpu
                batch_size = 512
                # Compute pitch using first gpu
                audio = torch.tensor(np.copy(x))[None].float()
                _, pd = torchcrepe.predict(
                    audio,
                    self.sr,
                    self.window,
                    f0_min,
                    f0_max,
                    model,
                    batch_size=batch_size,
                    device=self.device,
                    return_periodicity=True,
                )
                pd = torchcrepe.filter.median(pd, 3)
                pd = pd[0].cpu().numpy()
                pd = np.interp(
                    np.arange(0, len(pd) * f0.shape[0], len(pd)) / f0.shape[0],
                    np.arange(0, len(pd)),
                    pd,
                )
                f0[pd < 0.1] = 0

            if "privateuseone" in str(self.device):  # clean ortruntime memory
                del self.model_rmvpe.model
                del self.model_rmvpe
                logger.info("Cleaning ortruntime memory")
        elif f0_method == "fcpe":
            if not hasattr(self, "model_fcpe"):
                from torchfcpe import spawn_bundled_infer_model

                logger.info("Loading fcpe model")
                self.model_fcpe = spawn_bundled_infer_model(self.device)
            f0 = (
                self.model_fcpe.infer(
                    torch.from_numpy(x).to(self.device).unsqueeze(0).float(),
                    sr=16000,
                    decoder_mode="local_argmax",
                    threshold=0.006,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        if f0_invert_axis is not None:
            f0 = np.where(f0 >= f0_min, (f0_invert_axis**2) / f0, f0)

        f0 *= pow(2, f0_up_key / 12)
        # with open("test.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        tf0 = self.sr // self.window  # 每秒f0点数
        if inp_f0 is not None:
            delta_t = np.round(
                (inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1
            ).astype("int16")
            replace_f0 = np.interp(
                list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1]
            )
            shape = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[
                :shape
            ]
        # with open("test_opt.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)
        return f0_coarse, f0bak  # 1-0

    def vc(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        times,
        index,
        big_npy,
        index_rate,
        version,
        protect,
        block_length_override=None,
        feature_override=None,
        spec=None,
    ):  # ,file_index,file_big_npy
        t0 = ttime()
        if spec is not None:
            t1 = ttime()
            with torch.no_grad():
                if self.is_half:
                    spec = spec.half()
                else:
                    spec = spec.float()
                spec = spec.to(self.device)
                len_spec = torch.tensor([spec.shape[-1]], device=self.device).long()
                z, x_mask = net_g.get_hidden_features_q(
                    sid, spec.unsqueeze(0), len_spec
                )
                audio1 = (
                    (
                        net_g.infer_from_hidden_features(
                            sid,
                            pitchf,
                            z,
                            x_mask,
                        )[0, 0]
                    )
                    .data.cpu()
                    .float()
                    .numpy()
                )
                del spec, len_spec, z, x_mask
        else:
            feats = extract_features_simple_segment(
                audio0,
                model=model,
                version=version,
                device=self.device,
                is_half=self.is_half,
            )
            if pitch is not None and pitchf is not None:
                feats0 = feats.clone()
                feats_indexed = feats.clone()
            if not isinstance(index, type(None)) and not isinstance(
                big_npy, type(None)
            ):
                try:
                    npy = feats[0].cpu().numpy()
                    if self.is_half:
                        npy = npy.astype("float32")

                    # _, I = index.search(npy, 1)
                    # npy = big_npy[I.squeeze()]

                    score, ix = index.search(npy, k=8)
                    weight = np.square(1 / score)
                    weight /= weight.sum(axis=1, keepdims=True)
                    npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

                    if self.is_half:
                        npy = npy.astype("float16")
                    feats_indexed = torch.from_numpy(npy).unsqueeze(0).to(self.device)
                except:
                    print(traceback.format_exc())
                    feats_indexed = feats.clone()
                feats = feats_indexed * index_rate + (1 - index_rate) * feats

            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
            if pitch is not None and pitchf is not None:
                feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                    0, 2, 1
                )
                feats_indexed = F.interpolate(
                    feats_indexed.permute(0, 2, 1), scale_factor=2
                ).permute(0, 2, 1)
            t1 = ttime()
            p_len = audio0.shape[0] // self.window
            if feats.shape[1] < p_len:
                p_len = feats.shape[1]
                if pitch is not None and pitchf is not None:
                    pitch = pitch[:, :p_len]
                    pitchf = pitchf[:, :p_len]

            if (
                pitch is not None
                and pitchf is not None
                and not isinstance(index, type(None))
                and not isinstance(big_npy, type(None))
            ):
                pitchff = pitchf.clone()
                pitchff[pitchf > 0] = 1
                pitchff[pitchf < 1] = 0
                pitchff = pitchff.unsqueeze(-1)
                feats = feats * pitchff + (
                    feats0 * (1 - protect) + feats_indexed * protect
                ) * (1 - pitchff)
                feats = feats.to(feats0.dtype)
            if feature_override is not None:
                feats[:, :] = feature_override
            p_len = torch.tensor([p_len], device=self.device).long()
            with torch.no_grad():
                hasp = pitch is not None and pitchf is not None
                arg = (
                    (feats, p_len, pitch, pitchf, sid) if hasp else (feats, p_len, sid)
                )
                z, x_mask = net_g.get_hidden_features_p(
                    feats,
                    p_len,
                    pitch,
                    sid,
                    block_length_override=block_length_override,
                )
                if False:
                    f0_mel = 1127 * np.log(
                        1 + pitchf.squeeze(0).detach().cpu().numpy() / 700
                    )
                    f0 = (f0_mel - 550) / 120
                    z = z.squeeze(0).transpose(0, 1)

                    npy = z.detach().cpu().float().numpy()
                    if self.is_half:
                        npy = npy.astype("float32")

                    npy = np.concatenate(
                        (npy, f0[:, np.newaxis].astype(npy.dtype)), axis=1
                    )
                    npy[:, -1] *= 192**0.5

                    # _, I = index.search(npy, 1)
                    # npy = big_npy[I.squeeze()]

                    score, ix = index.search(npy, k=8)
                    weight = np.square(1 / score)
                    weight /= weight.sum(axis=1, keepdims=True)
                    npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

                    if self.is_half:
                        npy = npy.astype("float16")
                    npy = npy[:, :192]
                    z = torch.from_numpy(npy).to(self.device)
                    z = z.transpose(0, 1).unsqueeze(0)
                audio1 = (
                    (
                        net_g.infer_from_hidden_features(
                            sid,
                            pitchf,
                            z,
                            x_mask,
                        )[0, 0]
                    )
                    .data.cpu()
                    .float()
                    .numpy()
                )
                del hasp, arg
                del z, x_mask
            del feats, p_len
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        t2 = ttime()
        times[0] += t1 - t0
        times[2] += t2 - t1
        return audio1

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        input_audio_path,
        times,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version,
        protect,
        block_length_override=None,
        f0_invert_axis=None,
        feature_audio=None,
        if_feature_average=False,
        x_center_override=None,
        f0_file=None,
        f0_npy_path="",
        audio_40k=False,
    ):
        if (
            file_index != ""
            # and file_big_npy != ""
            # and os.path.exists(file_big_npy) == True
            and os.path.exists(file_index)
        ):
            try:
                index = faiss.read_index(file_index)
                # big_npy = np.load(file_big_npy)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except:
                traceback.print_exc()
                index = big_npy = None
        else:
            index = big_npy = None
        if if_feature_average:
            feats = extract_features_simple_segment(
                feature_audio,
                model=model,
                version=version,
                device=self.device,
                is_half=self.is_half,
            )
            npy = feats[0].cpu().numpy()
            npy = np.average(npy, axis=0)
            feature_override = torch.from_numpy(npy).to(self.device)
        else:
            feature_override = None
            if feature_audio is not None:
                if feature_audio.shape[0] > audio.shape[0]:
                    feature_audio = feature_audio[: audio.shape[0]]
                else:
                    feature_audio = np.pad(
                        feature_audio,
                        (0, audio.shape[0] - feature_audio.shape[0]),
                        mode="constant",
                    )
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        if feature_audio is not None:
            feature_audio = signal.filtfilt(bh, ah, feature_audio)
            feature_audio_pad = np.pad(
                feature_audio, (self.window // 2, self.window // 2), mode="reflect"
            )
        opt_ts = []
        t_center = (
            self.t_center
            if x_center_override is None
            else round(x_center_override * self.sr)
        )
        t_query = min(self.t_query, t_center // 2)
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += np.abs(audio_pad[i : i - self.window])
            for t in range(t_center, audio.shape[0], t_center):
                opt_ts.append(
                    t
                    - t_query
                    + np.where(
                        audio_sum[t - t_query : t + t_query]
                        == audio_sum[t - t_query : t + t_query].min()
                    )[0][0]
                )
        s = 0
        audio_opt = []
        t = 0
        t1 = ttime()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        if feature_audio is not None:
            feature_audio_pad = np.pad(
                feature_audio, (self.t_pad, self.t_pad), mode="reflect"
            )
        if audio_40k is not None:
            t_pad_40k = round(self.t_pad * len(audio_40k) / len(audio))
            audio_40k = np.pad(audio_40k, (t_pad_40k, t_pad_40k), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None
        if hasattr(f0_file, "name"):
            try:
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0 = []
                for line in lines:
                    inp_f0.append([float(i) for i in line.split(",")])
                inp_f0 = np.array(inp_f0, dtype="float32")
            except:
                traceback.print_exc()
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        if if_f0 == 1:
            pitch, pitchf = self.get_f0(
                input_audio_path,
                audio_pad,
                p_len,
                f0_up_key,
                f0_method,
                filter_radius,
                f0_invert_axis,
                inp_f0,
                f0_npy_path,
            )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if "mps" not in str(self.device) or "xpu" not in str(self.device):
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        if audio_40k is not None:
            audio_40k_torch = torch.FloatTensor(audio_40k.astype(np.float32)).unsqueeze(
                0
            )
            spec = spectrogram_torch(
                audio_40k_torch,
                2048,
                40000,
                400,
                2048,
                center=False,
            )
            spec = torch.squeeze(spec, 0)
            if spec.shape[-1] > p_len:
                spec = spec[..., :p_len]
            else:
                spec_pad = p_len - spec.shape[-1]
                spec = F.pad(spec, (0, spec_pad))
        if feature_audio is not None and not if_feature_average:
            audio, audio_pad = feature_audio, feature_audio_pad
        t2 = ttime()
        times[1] += t2 - t1
        for t in opt_ts:
            t = t // self.window * self.window
            if if_f0 == 1:
                if audio_40k is not None:
                    spec_slice = spec[
                        ..., s // self.window : (t + self.t_pad2) // self.window
                    ]
                else:
                    spec_slice = None
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                        pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                        times,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                        block_length_override=block_length_override,
                        feature_override=feature_override,
                        spec=spec_slice,
                    )[(self.t_pad_tgt - self.window) : -(self.t_pad_tgt - self.window)]
                )
            else:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        None,
                        None,
                        times,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                        feature_override=feature_override,
                    )[(self.t_pad_tgt - self.window) : -(self.t_pad_tgt - self.window)]
                )
            s = t
        if if_f0 == 1:
            if audio_40k is not None:
                spec_slice = spec[..., t // self.window :]
            else:
                spec_slice = None
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    pitch[:, t // self.window :],
                    pitchf[:, t // self.window :],
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                    feature_override=feature_override,
                    spec=spec_slice,
                )[(self.t_pad_tgt - self.window) : -(self.t_pad_tgt - self.window)]
            )
        else:
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    None,
                    None,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                    feature_override=feature_override,
                )[(self.t_pad_tgt - self.window) : -(self.t_pad_tgt - self.window)]
            )
        audio_opt = np.concatenate(audio_opt)
        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
        if tgt_sr != resample_sr >= 16000:
            audio_opt = librosa.resample(
                audio_opt, orig_sr=tgt_sr, target_sr=resample_sr
            )
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return audio_opt
