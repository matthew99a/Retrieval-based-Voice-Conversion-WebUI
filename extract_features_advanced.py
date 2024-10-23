import os
import sys
import traceback
import random
import shutil
from dotenv import load_dotenv

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

device = "cuda"
exp_dir = (
    "D:/matthew99/AI/singing_ai/Retrieval-based-Voice-Conversion-WebUI/logs/mi-test"
)
import fairseq
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

now_dir = os.getcwd()
sys.path.append(now_dir)
from infer.lib.audio import load_audio
from infer.lib.audio import extract_features_simple_segment

from infer.modules.vc.modules import VC
from configs.config import Config
from scipy.io import wavfile

from scipy.interpolate import CubicSpline

eps = 1e-3

config = Config()
vc = VC(config)

from f0_magic import resize_with_zeros
from infer.lib.audio import pitch_blur

version = "v2"

model_path = "assets/hubert/hubert_base.pt"
# HuBERT model
print("load model(s) from {}".format(model_path))
# if hubert model is exist
if os.access(model_path, os.F_OK) == False:
    print(
        "Error: Extracting is shut down because %s does not exist, you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
        % model_path
    )
    exit(0)
models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    [model_path],
    suffix="",
)
model = models[0]
model = model.to(device)
print("move model to %s" % device)
if config.is_half:
    if device not in ["mps", "cpu"]:
        model = model.half()
model.eval()


model_rmvpe = None


def compute_f0(path):
    print("computing f0 for: " + path)
    x = load_audio(path, 16000)

    global model_rmvpe
    if model_rmvpe is None:
        from infer.lib.rmvpe import RMVPE

        print("Loading rmvpe model")
        model_rmvpe = RMVPE("assets/rmvpe/rmvpe.pt", is_half=False, device="cuda")
    f0 = model_rmvpe.infer_from_audio(x, thred=0.03)
    return f0


def add_noise(contour, amp=5, scale=1):
    zeros = contour < 0.001
    contour = 1127 * np.log(1 + contour / 700)
    length = int(contour.shape[0] / scale) + 1
    noise = np.random.normal(0, amp, length)
    if len(noise) != len(contour):
        noise = CubicSpline(np.arange(0, len(noise)), noise)(
            np.arange(0, len(noise) * len(contour), len(noise)) / len(contour)
        )
    contour_with_noise = contour + noise
    contour_with_noise = (np.exp(contour_with_noise / 1127) - 1) * 700
    contour_with_noise[zeros] = 0
    return contour_with_noise


vc_list = (
    ("0", 0),
    ("1", 0),
    ("2", 0),
    ("3", -9),
    ("4", -9),
    ("5", -9),
    ("6", -12),
    ("7", -15),
    (None, 0),
)

model_path = "assets/hubert/hubert_base.pt"

modelsPath = "D:/matthew99/AI/singing_ai/downloaded"
wavPath = "%s/0_gt_wavs" % exp_dir
outPath = "%s/3_feature768" % exp_dir
os.makedirs(outPath, exist_ok=True)

todo = sorted(list(os.listdir(wavPath)))
vc_todos = [[] for i in range(len(vc_list))]
total = 0
for idx, file in enumerate(todo):
    if file.endswith(".wav"):
        total += 1
        vc_todos[random.randint(0, len(vc_list) - 1)].append(file)

idx = 0
n = max(1, total // 100)

for i in range(len(vc_list)):
    vc_name, shift = vc_list[i]
    pth_path = ""
    index_path = ""
    if vc_name is not None:
        for file in os.listdir("%s/%s" % (modelsPath, vc_name)):
            full_path = "%s/%s/%s" % (modelsPath, vc_name, file)
            if file.endswith(".pth"):
                pth_path = full_path
            elif file.endswith(".index"):
                index_path = full_path
    vc.get_vc(pth_path)
    for file in vc_todos[i]:
        try:
            idx += 1
            wav_path = "%s/%s" % (wavPath, file)
            out_path = "%s/%s" % (outPath, file.replace("wav", "npy"))
            f0_npy_path = "%s/%s" % (outPath, file + ".npy")
            try:
                np.load(out_path)
            except:
                if vc_name is None:
                    audio = load_audio(wav_path, 16000)
                else:
                    #                    f0 = add_noise(
                    #                        pitch_blur(compute_f0(wav_path)),
                    #                        amp=random.uniform(0, 20),
                    #                        scale=random.randint(3, 10),
                    #                    )
                    f0 = compute_f0(wav_path)
                    f0 = f0 * (2 ** ((shift - random.uniform(0, 3)) / 12))
                    f0 = np.pad(f0, (300, 300))
                    np.save(f0_npy_path, f0, allow_pickle=False)
                    sr, opt = vc.vc_single(
                        0,
                        wav_path,
                        0,
                        None,
                        "rmvpe",
                        index_path,
                        "",
                        0 if index_path == "" else 1,
                        3,
                        16000,
                        0,
                        0,
                        f0_npy_path=f0_npy_path,
                        output_to_file=False,
                    )[1]
                    os.remove(f0_npy_path)
                    audio = opt / max(np.abs(opt).max(), 32768)

                feats = extract_features_simple_segment(
                    audio, model=model, version=version, device=device
                )

                feats = feats.squeeze(0).float().cpu().numpy()

                if np.isnan(feats).sum() == 0:
                    np.save(out_path, feats, allow_pickle=False)
                else:
                    print("%s-contains nan" % file)
                if idx % n == 0:
                    print("now-%s,all-%s,%s,%s" % (idx, total, file, feats.shape))
        except:
            print(traceback.format_exc())
