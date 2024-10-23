import traceback
import logging

logger = logging.getLogger(__name__)

import numpy as np
import soundfile as sf
import torch
from io import BytesIO

from infer.lib.audio import load_audio, wav2, clean_path
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.modules.vc.pipeline import Pipeline
from infer.modules.vc.utils import *
from infer.lib.train.mel_processing import spectrogram_torch

import os, time

ENC_Q = False


class VC:
    def __init__(self, config):
        self.n_spk = None
        self.tgt_sr = None
        self.net_g = None
        self.pipeline = None
        self.cpt = None
        self.version = None
        self.if_f0 = None
        self.version = None
        self.hubert_model = None
        self.enc_q = ENC_Q

        self.config = config

    def get_vc(self, sid, *to_return_protect):
        logger.info("Get sid: " + sid)

        to_return_protect0 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[0] if self.if_f0 != 0 and to_return_protect else 1.0
            ),
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[1] if self.if_f0 != 0 and to_return_protect else 0.67
            ),
            "__type__": "update",
        }

        if sid == "" or sid == []:
            if (
                self.hubert_model is not None
            ):  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
                logger.info("Clean model cache")
                del (self.net_g, self.n_spk, self.hubert_model, self.tgt_sr)  # ,cpt
                self.hubert_model = self.net_g = self.n_spk = self.hubert_model = (
                    self.tgt_sr
                ) = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                ###楼下不这么折腾清理不干净
                self.if_f0 = self.cpt.get("f0", 1)
                self.version = self.cpt.get("version", "v1")
                if self.version == "v1":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs256NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs256NSFsid_nono(*self.cpt["config"])
                elif self.version == "v2":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs768NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs768NSFsid_nono(*self.cpt["config"])
                del self.net_g, self.cpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            return (
                {"visible": False, "__type__": "update"},
                to_return_protect0,
                to_return_protect1,
                "",
                "",
            )
        if os.path.isfile(sid):
            person = sid
        else:
            person = f'{os.getenv("weight_root")}/{sid}'
        logger.info(f"Loading: {person}")

        self.cpt = torch.load(person, map_location="cpu")
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }

        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*self.cpt["config"], is_half=self.config.is_half)

        if not self.enc_q:
            del self.net_g.enc_q

        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        if hasattr(self.net_g, "enc_p2"):
            keep = False
            for key in self.cpt["weight"].keys():
                if "enc_p2." in key:
                    keep = True
                    break
            if not keep:
                del self.net_g.enc_p2
        self.net_g.eval().to(self.config.device)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)
        n_spk = self.cpt["config"][-3]
        try:
            index = {"value": get_index_path_from_model(sid), "__type__": "update"}
        except:
            index = {"value": "", "__type__": "update"}
        logger.info("Select index: " + index["value"])

        return (
            (
                {"visible": True, "maximum": n_spk, "__type__": "update"},
                to_return_protect0,
                to_return_protect1,
                index,
                index,
            )
            if to_return_protect
            else {"visible": True, "maximum": n_spk, "__type__": "update"}
        )

    def get_hidden_features(
        self,
        sid,
        input_audio_path,
    ):
        input_audio_path = clean_path(input_audio_path)
        audio_40k = load_audio(input_audio_path, 40000)
        audio_40k_max = np.abs(audio_40k).max() / 0.95
        if audio_40k_max > 1:
            audio_40k /= audio_40k_max
        audio_40k_torch = torch.FloatTensor(audio_40k.astype(np.float32)).unsqueeze(0)
        spec = spectrogram_torch(
            audio_40k_torch,
            2048,
            40000,
            400,
            2048,
            center=False,
        )
        spec = torch.squeeze(spec, 0)
        if self.config.is_half:
            spec = spec.half()
        else:
            spec = spec.float()
        sid = torch.tensor(sid, device=self.config.device).unsqueeze(0).long()
        spec = spec.to(self.config.device)
        len_spec = torch.LongTensor(1).to(self.config.device)
        len_spec[0] = spec.shape[-1]
        feats = self.net_g.get_hidden_features_q(
            sid,
            y=spec.unsqueeze(0),
            y_lengths=len_spec,
        )[0].data.squeeze(0)
        feats = feats[:, : spec.shape[-1]]
        del sid, spec, len_spec
        feats = feats.transpose(0, 1)
        feats = feats.cpu().float().numpy()
        return feats

    def vc_single(
        self,
        sid,
        input_audio_path,
        f0_up_key,
        f0_file,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        block_length_override_=0,
        f0_invert_axis=" ",
        feature_audio_path="",
        if_feature_average=False,
        segment_length=None,
        f0_npy_path="",
        output_to_file=True,
    ):
        if input_audio_path is None:
            return "You need to upload an audio", None
        if feature_audio_path != "":
            feature_audio_path = clean_path(feature_audio_path)
        if f0_npy_path != "":
            f0_npy_path = clean_path(f0_npy_path)
        input_audio_path = clean_path(
            input_audio_path
        )  # 防止小白拷路径头尾带了空格和"和回车
        f0_up_key = int(f0_up_key)
        if block_length_override_ == 0:
            block_length_override = None
        else:
            block_length_override = int(round(block_length_override_ * 100))

        if f0_invert_axis != " ":
            notes = [
                "C2",
                "C#2/Db2",
                "D2",
                "D#2/Eb2",
                "E2",
                "F2",
                "F#2/Gb2",
                "G2",
                "G#2/Ab2",
                "A2",
                "A#2/Bb2",
                "B2",
                "C3",
                "C#3/Db3",
                "D3",
                "D#3/Eb3",
                "E3",
                "F3",
                "F#3/Gb3",
                "G3",
                "G#3/Ab3",
                "A3",
                "A#3/Bb3",
                "B3",
                "C4",
                "C#4/Db4",
                "D4",
                "D#4/Eb4",
                "E4",
                "F4",
                "F#4/Gb4",
                "G4",
                "G#4/Ab4",
                "A4",
                "A#4/Bb4",
                "B4",
                "C4",
                "C#4/Db4",
                "D4",
                "D#4/Eb4",
                "E4",
                "F4",
                "F#4/Gb4",
                "G4",
                "G#4/Ab4",
                "A4",
                "A#4/Bb4",
                "B4",
                "C5",
                "C#5/Db5",
                "D5",
                "D#5/Eb5",
                "E5",
                "F5",
                "F#5/Gb5",
                "G5",
                "G#5/Ab5",
                "A5",
                "A#5/Bb5",
                "B5",
            ]
            offset = -33
            for note in notes:
                if note == f0_invert_axis:
                    break
                offset += 1
            f0_invert_axis = 440 * (2 ** (offset / 12))
        else:
            f0_invert_axis = None

        try:
            audio = load_audio(input_audio_path, 16000)
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            if ENC_Q:
                audio_40k = load_audio(input_audio_path, 40000)
                audio_40k_max = np.abs(audio_40k).max() / 0.95
                if audio_40k_max > 1:
                    audio_40k /= audio_40k_max
            else:
                audio_40k = None

            if feature_audio_path != "":
                feature_audio = load_audio(feature_audio_path, 16000)
                feature_audio_max = np.abs(feature_audio).max() / 0.95
                if feature_audio_max > 1:
                    feature_audio /= feature_audio_max
            else:
                feature_audio = None

            times = [0, 0, 0]

            if self.hubert_model is None:
                self.hubert_model = load_hubert(self.config)

            if file_index:
                file_index = (
                    file_index.strip(" ")
                    .strip('"')
                    .strip("\n")
                    .strip('"')
                    .strip(" ")
                    .replace("trained", "added")
                )
            elif file_index2:
                file_index = file_index2
            else:
                file_index = ""  # 防止小白写错，自动帮他替换掉

            audio_opt = self.pipeline.pipeline(
                self.hubert_model,
                self.net_g,
                sid,
                audio,
                input_audio_path,
                times,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                self.if_f0,
                filter_radius,
                self.tgt_sr,
                resample_sr,
                rms_mix_rate,
                self.version,
                protect,
                block_length_override=block_length_override,
                f0_invert_axis=f0_invert_axis,
                feature_audio=feature_audio,
                if_feature_average=if_feature_average,
                x_center_override=segment_length,
                f0_file=f0_file,
                f0_npy_path=f0_npy_path,
                audio_40k=audio_40k,
            )
            if self.tgt_sr != resample_sr >= 16000:
                tgt_sr = resample_sr
            else:
                tgt_sr = self.tgt_sr
            index_info = (
                "Index:\n%s." % file_index
                if os.path.exists(file_index)
                else "Index not used."
            )
            if output_to_file:
                try:
                    sf.write(
                        "D:/matthew99/AI/singing_ai/tmp/output %s %.0f %.2f %.2f %.2f %.2f %.2f %s %f.wav"
                        % (
                            os.path.splitext(os.path.split(input_audio_path)[1])[0],
                            f0_up_key,
                            index_rate,
                            protect,
                            segment_length,
                            rms_mix_rate,
                            block_length_override_,
                            f0_method,
                            time.time() * 1000,
                        ),
                        audio_opt,
                        tgt_sr,
                    )
                    print(
                        "Success.\n%s\nTime:\nnpy: %.2fs, f0: %.2fs, infer: %.2fs."
                        % (index_info, *times)
                    )
                except:
                    pass
            return (
                "Success.\n%s\nTime:\nnpy: %.2fs, f0: %.2fs, infer: %.2fs."
                % (index_info, *times),
                (tgt_sr, audio_opt),
            )
        except:
            info = traceback.format_exc()
            logger.warning(info)
            return info, (None, None)

    def vc_multi(
        self,
        sid,
        dir_path,
        opt_root,
        paths,
        f0_up_key,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        format1,
    ):
        try:
            dir_path = clean_path(dir_path)  # 防止小白拷路径头尾带了空格和"和回车
            opt_root = clean_path(opt_root)
            os.makedirs(opt_root, exist_ok=True)
            try:
                if dir_path != "":
                    paths = [
                        os.path.join(dir_path, name) for name in os.listdir(dir_path)
                    ]
                else:
                    paths = [path.name for path in paths]
            except:
                traceback.print_exc()
                paths = [path.name for path in paths]
            infos = []
            for path in paths:
                info, opt = self.vc_single(
                    sid,
                    path,
                    f0_up_key,
                    None,
                    f0_method,
                    file_index,
                    file_index2,
                    # file_big_npy,
                    index_rate,
                    filter_radius,
                    resample_sr,
                    rms_mix_rate,
                    protect,
                )
                if "Success" in info:
                    try:
                        tgt_sr, audio_opt = opt
                        if format1 in ["wav", "flac"]:
                            sf.write(
                                "%s/%s.%s"
                                % (opt_root, os.path.basename(path), format1),
                                audio_opt,
                                tgt_sr,
                            )
                        else:
                            path = "%s/%s.%s" % (
                                opt_root,
                                os.path.basename(path),
                                format1,
                            )
                            with BytesIO() as wavf:
                                sf.write(wavf, audio_opt, tgt_sr, format="wav")
                                wavf.seek(0, 0)
                                with open(path, "wb") as outf:
                                    wav2(wavf, outf, format1)
                    except:
                        info += traceback.format_exc()
                infos.append("%s->%s" % (os.path.basename(path), info))
                yield "\n".join(infos)
            yield "\n".join(infos)
        except:
            yield traceback.format_exc()
