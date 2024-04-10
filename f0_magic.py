import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
import os

import librosa
import ffmpeg

from infer.lib.audio import load_audio, pitch_blur_mel, extract_features_simple, trim_sides_mel
import torchcrepe
import random

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.interpolate import CubicSpline
from scipy.signal import medfilt 

from configs.config import Config
from infer.modules.vc.utils import load_hubert
from f0_magic_gen import PitchContourGenerator, segment_size
from f0_magic_disc import PitchContourDiscriminator 
from f0_magic_disc import segment_size as segment_size_disc

config = Config()

eps = 1e-3
mel_min = 1127 * math.log(1 + 50 / 700)
mel_max = 1127 * math.log(1 + 1100 / 700)

multiplicity_target = 40
multiplicity_others = 40
max_offset = round(segment_size / 40)
min_ratio = 0.5
median_filter_size = 17
gaussian_filter_sigma = 8
data_noise_amp = 5
label_noise_amp = 0.1

USE_TEST_SET = True
EPOCH_PER_BAK = 25

lr_g = 1e-5
lr_d = 1e-5
c_loss_factor = 0.1

mn = 559.4985610615364
std = 120.52172592468257
def preprocess(x):
    x_ret = x.clone()
    x_ret = (x_ret - mn) / std
#    x_ret[x < eps] = (2 * mel_min - mel_max - mn) / std
    return x_ret


def postprocess(x):
    x_ret = x.clone()
    x_ret = x * std + mn
    x_ret[x_ret < mel_min * 0.5] = 0
#    x_ret[x < eps] = (2 * mel_min - mel_max - mn) / std
    return x_ret


sr = 16000
window_length = 160
frames_per_sec = sr // window_length
def resize_with_zeros(contour, target_len):
    a = contour.copy()
    a[a < eps] = np.nan
    a = np.interp(
        np.arange(0, len(a) * target_len, len(a)) / target_len,
        np.arange(0, len(a)),
        a
    )
    a = np.nan_to_num(a)
    return a


hubert_model = None
def trim_f0(f0, audio, index_file, version="v2"):
    global hubert_model

    if not os.path.isfile(index_file):
        return f0
    import faiss
    try:
        index = faiss.read_index(index_file)
        # big_npy = np.load(file_big_npy)
        big_npy = index.reconstruct_n(0, index.ntotal)
    except:
        print("Failed to read index file: \"{index_file:s}\"")
        return f0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hubert_model is None:
        hubert_model = load_hubert(config)

    feats = extract_features_simple(audio, model=hubert_model, version=version, device=device, is_half=config.is_half)
    npy = feats[0].cpu().numpy()
    npy = np.concatenate((npy, np.full((npy.shape[0], 1), 0.5)), axis=1)

    score, ix = index.search(npy, k=8)
    weight = np.square(1 / score)
    weight /= weight.sum(axis=1, keepdims=True)
    npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

    pd = npy[:, -1]
    pd = np.interp(
        np.arange(0, len(pd) * len(f0), len(pd)) / len(f0),
        np.arange(0, len(pd)),
        pd
    )

    threshold = 0.5
    for it in (range(len(f0)), reversed(range(len(f0)))):
        keep = False
        for i in it:
            if f0[i] > eps:
                if pd[i] > threshold:
                    keep = True
                if not keep:
                    f0[i] = 0
            else:
                keep = False

    return f0


model_rmvpe = None
def compute_f0_inference(path, index_file=""):
    print("computing f0 for: " + path)
    x = load_audio(path, 44100)
    x = librosa.resample(
        x, orig_sr=44100, target_sr=sr
    )

    global model_rmvpe
    if model_rmvpe is None:
        from infer.lib.rmvpe import RMVPE
        print("Loading rmvpe model")
        model_rmvpe = RMVPE(
            "assets/rmvpe/rmvpe.pt", is_half=False, device="cuda")
    f0 = model_rmvpe.infer_from_audio(x, thred=0.03)

    # Pick a batch size that doesn't cause memory errors on your gpu
    torch_device_index = 0
    torch_device = None
    if torch.cuda.is_available():
        torch_device = torch.device(f"cuda:{torch_device_index % torch.cuda.device_count()}")
    elif torch.backends.mps.is_available():
        torch_device = torch.device("mps")
    else:
        torch_device = torch.device("cpu")
    model = "full"
    batch_size = 512
    # Compute pitch using first gpu
    audio_tensor = torch.tensor(np.copy(x))[None].float()
    f0_crepe, pd = torchcrepe.predict(
        audio_tensor,
        16000,
        160,
        50,
        1100,
        model,
        batch_size=batch_size,
        device=torch_device,
        return_periodicity=True,
    )
    pd = torchcrepe.filter.median(pd, 3)
    f0_crepe = torchcrepe.filter.mean(f0_crepe, 3)
    f0_crepe[pd < 0.1] = 0
    f0_crepe = f0_crepe[0].cpu().numpy()
    f0_crepe = f0_crepe[1:] # Get rid of extra first frame

    # Resize the pitch
    target_len = f0.shape[0]
    f0_crepe = resize_with_zeros(f0_crepe, target_len)

    f0_rmvpe_mel = np.log(1 + f0 / 700)
    f0_crepe_mel = np.log(1 + f0_crepe / 700)
    f0 = np.where(np.logical_and(f0_rmvpe_mel > eps, f0_crepe_mel - f0_rmvpe_mel > 0.05), f0_crepe, f0)

    f0_mel = 1127 * np.log(1 + f0 / 700)

    target_len = x.shape[0] // window_length
    f0_mel = resize_with_zeros(f0_mel, target_len)

    if index_file != "":
        f0_mel = trim_f0(f0_mel, x, index_file)

    f0_mel = trim_sides_mel(f0_mel, frames_per_sec)

    f0 = (np.exp(f0_mel / 1127) - 1) * 700 
    f0 = np.pad(f0, (300, 300))
    return f0


model_rmvpe = None
def compute_f0(path):
    print("computing f0 for: " + path)
    x = load_audio(path, 44100)
    x = librosa.resample(
        x, orig_sr=44100, target_sr=sr
    )

    global model_rmvpe
    if model_rmvpe is None:
        from infer.lib.rmvpe import RMVPE
        print("Loading rmvpe model")
        model_rmvpe = RMVPE(
            "assets/rmvpe/rmvpe.pt", is_half=False, device="cuda")
    f0 = model_rmvpe.infer_from_audio(x, thred=0.03)

    # Pick a batch size that doesn't cause memory errors on your gpu
    torch_device_index = 0
    torch_device = None
    if torch.cuda.is_available():
        torch_device = torch.device(f"cuda:{torch_device_index % torch.cuda.device_count()}")
    elif torch.backends.mps.is_available():
        torch_device = torch.device("mps")
    else:
        torch_device = torch.device("cpu")
    model = "full"
    batch_size = 512
    # Compute pitch using first gpu
    audio_tensor = torch.tensor(np.copy(x))[None].float()
    f0_crepe, pd = torchcrepe.predict(
        audio_tensor,
        16000,
        160,
        50,
        1100,
        model,
        batch_size=batch_size,
        device=torch_device,
        return_periodicity=True,
    )
    pd = torchcrepe.filter.median(pd, 3)
    f0_crepe = torchcrepe.filter.mean(f0_crepe, 3)
    f0_crepe[pd < 0.1] = 0
    f0_crepe = f0_crepe[0].cpu().numpy()
    f0_crepe = f0_crepe[1:] # Get rid of extra first frame

    # Resize the pitch
    target_len = f0.shape[0]
    f0_crepe = resize_with_zeros(f0_crepe, target_len)

    f0_rmvpe_mel = np.log(1 + f0 / 700)
    f0_crepe_mel = np.log(1 + f0_crepe / 700)
    f0 = np.where(np.logical_and(f0_rmvpe_mel > eps, f0_crepe_mel - f0_rmvpe_mel > 0.05), f0_crepe, f0)

    f0_mel = 1127 * np.log(1 + f0 / 700)

    target_len = x.shape[0] // window_length
    f0_mel = resize_with_zeros(f0_mel, target_len)
    return f0_mel



TARGET_PATH = "f0_magic/target"
OTHERS_PATH = "f0_magic/others"

def walk(path):
   return sum(([os.path.join(dirpath, file_name) for file_name in filenames] for (dirpath, dirnames, filenames) in os.walk(path)), [])


def prepare_data():
    filenames = []
    for filename in walk(TARGET_PATH):
        if filename.endswith(".wav"): 
            filenames.append(filename)
    for filename in walk(OTHERS_PATH):
        if filename.endswith(".wav"):
            filenames.append(filename)
    for filename in filenames:
        npy_file = os.path.splitext(filename)[0] + ".npy"
        if not os.path.isfile(npy_file):
            try:
                np.save(npy_file, compute_f0(filename))
            except:
                os.remove(filename)


def pitch_shift_mel(contour, semitones):
    contour = (np.exp(contour / 1127) - 1) * 700
    contour *= 2 ** (semitones / 12)
    contour = 1127 * np.log(1 + contour / 700)
    contour[contour < eps] = 0
    return contour


def pitch_invert_mel(contour, note):
    contour = (np.exp(contour / 1127) - 1) * 700
    contour[contour > 0] = (librosa.note_to_hz(note) ** 2) / contour[contour > 0]
    contour = 1127 * np.log(1 + contour / 700)
    contour[contour < eps] = 0
    return contour


def add_noise(contour, amp=5, scale=1):
    zeros = contour < eps
    length = int(contour.shape[0] / scale) + 1
    noise = np.random.normal(0, amp, length)
    if len(noise) != len(contour):
        noise = CubicSpline(np.arange(0, len(noise)), noise)(np.arange(0, len(noise) * len(contour), len(noise)) / len(contour))
    contour_with_noise = contour + noise
    contour_with_noise[zeros] = 0
    return contour_with_noise


def get_average(contour):
    try:
        return np.average(contour[contour > eps])
    except ZeroDivisionError:
        return 0


def change_vibrato(contour, factor):
    blurred = pitch_blur_mel(contour, frames_per_sec)
    modified_contour = blurred + factor * (contour - blurred)
    modified_contour[modified_contour < eps] = 0
    return modified_contour


def modify_ends(contour):
    from scipy.ndimage import gaussian_filter1d
    contour_pad = np.concatenate(([0], contour))
    contour_segments = np.split(contour_pad, np.where(contour_pad < eps)[0])
    border_length = random.randint(4, 24)
    amount = random.uniform(30, 60) * random.choice((-1, 1))
    t = random.randint(0, 1)
    mask = np.hanning(border_length * 2)
    if t == 0:
        mask = mask[border_length:]
    else:
        mask = mask[:border_length]
    mask *= amount
    modified_segments = []
    for segment in contour_segments:
        if segment.shape[0] > 0:
            if len(segment) > border_length:
                if t == 0:
                    segment[1:border_length + 1] += mask
                else:
                    segment[-border_length:] += mask
            modified_segments.append(segment)
    modified_contour = np.concatenate(modified_segments)[1:]
    return modified_contour


def load_data():
    prepare_data()
    train_target_data = []
    train_others_data = []
    test_target_data = []
    test_others_data = []
    test_set = set()
    for filename in walk(TARGET_PATH) + walk(OTHERS_PATH):
        if filename.endswith(".npy"): 
            if random.uniform(0, 1) < 0.2:
                test_set.add(filename)
    for filename in walk(TARGET_PATH):
        if filename.endswith(".npy"): 
            if filename in test_set:
                target_data, others_data = test_target_data, test_others_data
            else:
                target_data, others_data = train_target_data, train_others_data
                contour = np.load(filename)
            if contour.shape[0] < segment_size:
                contour = np.pad(contour, (0, segment_size - contour.shape[0]))
            contour = np.pad(contour, (segment_size + max_offset, segment_size))
            for i in range(int(multiplicity_target * (contour.shape[0] - 3 * segment_size - max_offset) / segment_size) + 1):
                start = random.randint(segment_size + max_offset, contour.shape[0] - segment_size * 2)
                use_original = random.randint(0, 4) == 0
                contour_sliced = contour[start - segment_size - max_offset:start + 2 * segment_size].copy()
                if np.sum(contour_sliced[segment_size:-segment_size] > eps) > segment_size * min_ratio:
                    contour_final = contour_sliced
                    target_data.append(torch.tensor(contour_final, dtype=torch.float32))

    if multiplicity_others > 0:
        for filename in walk(OTHERS_PATH):
            if filename.endswith(".npy"):
                if filename in test_set:
                    target_data, others_data = test_target_data, test_others_data
                else:
                    target_data, others_data = train_target_data, train_others_data
                contour = np.load(filename)
                if contour.shape[0] < segment_size:
                    contour = np.pad(contour, (0, segment_size - contour.shape[0]))
                contour = np.pad(contour, (segment_size + max_offset, segment_size))
                for i in range(int(multiplicity_others * (contour.shape[0] - 3 * segment_size - max_offset) / segment_size) + 1):
                    start = random.randint(segment_size + max_offset, contour.shape[0] - segment_size * 2)
                    use_original = random.randint(0, 4) == 0
                    contour_sliced = contour[start - segment_size - max_offset:start + 2 * segment_size].copy()
                    if np.sum(contour_sliced[segment_size:-segment_size] > eps) > segment_size * min_ratio:
                        if use_original:
                            shift_real = 0
                        else:
                            shift = random.uniform(0, 1)
                            average = get_average(contour_sliced[contour_sliced > eps]) / 1127
                            LOW, HIGH = librosa.note_to_hz("Ab3"), librosa.note_to_hz("Bb5")
                            LOW, HIGH = math.log(1 + LOW / 700), math.log(1 + HIGH / 700)
                            average_goal = (HIGH - LOW) * shift + LOW
                            average = (math.exp(average) - 1) * 700
                            average_goal = (math.exp(average_goal) - 1) * 700
                            shift_real = math.log(average_goal / average) / math.log(2) * 12
                        contour_final = pitch_shift_mel(contour_sliced, shift_real)
                        others_data.append(torch.tensor(contour_final, dtype=torch.float32))

    print("Train target data count:", len(train_target_data))
    print("Train others data count:", len(train_others_data))
    print("Test target data count:", len(test_target_data))
    print("Test others data count:", len(test_others_data))
    return train_target_data, train_others_data, test_target_data, test_others_data


def median_filter1d_torch(x, size):
    return torch.median(torch.cat(tuple(x[:, i:x.shape[1] - size + i + 1].unsqueeze(2) for i in range(size)), dim=2), dim=2).values


def gaussian_filter1d_torch(x, sigma, width=None):
    if width is None:
        width = round(sigma * 4)
    distance = torch.arange(
        -width, width + 1, dtype=torch.float32, device=x.device
    )
    gaussian = torch.exp(
        -(distance ** 2) / (2 * sigma ** 2)
    )
    gaussian /= gaussian.sum()
    kernel = gaussian[None, None].expand(1, -1, -1)
    return F.conv1d(x.unsqueeze(1), kernel, padding="same").squeeze(1)


def contrastive_loss(output, ref, size):
#    output = gaussian_filter1d_torch(output, size)
#    ref = gaussian_filter1d_torch(ref, size)
    output = output[:, segment_size:-segment_size]
    ref = ref[:, segment_size:-segment_size]
    return F.mse_loss(output, ref)


def train_model(name, train_target_data, train_others_data, test_target_data, test_others_data):
    if train_target_data:
        train_target_data = torch.stack(train_target_data)
    if train_others_data:
        train_others_data = torch.stack(train_others_data)
    if test_target_data:
        test_target_data = torch.stack(test_target_data)
    if test_others_data:
        test_others_data = torch.stack(test_others_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_g = PitchContourGenerator().to(device)
    net_d = PitchContourDiscriminator().to(device)
    optimizer_g = optim.AdamW(net_g.parameters(), lr=lr_g)
    optimizer_d = optim.AdamW(net_d.parameters(), lr=lr_d)
    epoch = 0

    MODEL_FILE = name + ".pt"
    CHECKPOINT_FILE = name + " checkpoint.pt"
    TMP_FILE = name + "_tmp.pt"
    if os.path.isfile(TMP_FILE):
        if not os.path.isfile(CHECKPOINT_FILE):
            os.rename(TMP_FILE, CHECKPOINT_FILE)
        else:
            os.remove(TMP_FILE)
    try:
        if os.path.isfile(CHECKPOINT_FILE):
            checkpoint = torch.load(CHECKPOINT_FILE)
            epoch = checkpoint['epoch']
            net_g.load_state_dict(checkpoint['net_g'])
            net_d.load_state_dict(checkpoint['net_d'])
            optimizer_g.load_state_dict(checkpoint['optimizer_g'])
            optimizer_d.load_state_dict(checkpoint['optimizer_d'])
            print(f"Data loaded from '{CHECKPOINT_FILE:s}'")
        else:
            print("Model initialized with random weights")
    except:
        epoch = 0
        net_g = PitchContourGenerator().to(device)
        net_d = PitchContourDiscriminator().to(device)
        optimizer_g = optim.Adam(net_g.parameters(), lr=lr_g)
        optimizer_d = optim.Adam(net_d.parameters(), lr=lr_d)
        print("Model initialized with random weights")

    train_dataset = torch.utils.data.TensorDataset(train_target_data, torch.ones((len(train_target_data),)))
    if len(train_others_data):
        train_dataset += torch.utils.data.TensorDataset(train_others_data, torch.zeros((len(train_others_data),)))
    if USE_TEST_SET:
        test_dataset = torch.utils.data.TensorDataset(test_target_data, torch.ones((len(test_target_data),)))
        if len(test_others_data):
            test_dataset += torch.utils.data.TensorDataset(test_others_data, torch.zeros((len(test_others_data),)))
    else:
        if len(test_target_data):
            train_dataset += torch.utils.data.TensorDataset(test_target_data, torch.ones((len(test_target_data),)))
        if len(test_others_data):
            train_dataset += torch.utils.data.TensorDataset(test_others_data, torch.zeros((len(test_others_data),)))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    if USE_TEST_SET:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=True)

    criterion = nn.BCELoss()

    net_g.train()
    net_d.train()

    best_loss = float("inf")

    while True:
        epoch += 1

        train_disc_loss = 0
        train_contrastive_loss = 0
        train_gen_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            offset = torch.randint(0, max_offset, (1,))
            data = data[:, offset:data.shape[1] - max_offset + offset]
#            data_disturbed = data + torch.randn_like(data) * data_noise_amp

            fakes = postprocess(net_g(preprocess(data.unsqueeze(1)))).squeeze(1)
#            fakes[data < eps] = 0
            d_data = fakes.detach().clone()
            d_data[labels > eps] = data[labels > eps]
#            d_data = d_data + torch.randn_like(d_data) * data_noise_amp
            d_labels = labels
#            d_labels = labels * (1 - label_noise_amp) + torch.rand(size=labels.shape, device=device) * label_noise_amp
#            d_labels = torch.zeros((d_data.shape[0],), device=device)
            if torch.sum(labels > eps) > 0:
                target_data = data[labels > eps] + torch.randn_like(data[labels > eps]) * data_noise_amp
                target_labels = torch.zeros((target_data.shape[0],), device=device)
                d_data = torch.cat((d_data, target_data), dim=0)
                d_labels = torch.cat((d_labels, target_labels), dim=0)
            d_data = d_data[:, (d_data.shape[1] + segment_size_disc) // 2 - segment_size_disc:(d_data.shape[1] + segment_size_disc) // 2]
            outputs = net_d(preprocess(d_data.unsqueeze(1)))

            optimizer_d.zero_grad()
            loss = criterion(outputs, d_labels.unsqueeze(1))
            loss.backward()
            optimizer_d.step()

            train_disc_loss += loss.item()


            if torch.sum(labels < eps) > 0:
                g_data = fakes[labels < eps]
                g_labels = torch.ones((g_data.shape[0],), device=device)
                g_data = g_data[:, (g_data.shape[1] + segment_size_disc) // 2 - segment_size_disc:(g_data.shape[1] + segment_size_disc) // 2]
                outputs = net_d(preprocess(g_data.unsqueeze(1)))

                loss_total = 0
                loss = criterion(outputs, g_labels.unsqueeze(1))
                loss_total = loss
                train_gen_loss += loss.item()

                loss = contrastive_loss(fakes, data, gaussian_filter_sigma)
                loss_total += loss * c_loss_factor
                train_contrastive_loss += loss.item()

                optimizer_g.zero_grad()
                loss_total.backward()
                optimizer_g.step()

        train_disc_loss /= len(train_loader)
        train_contrastive_loss /= len(train_loader)
        train_gen_loss /= len(train_loader)
        train_loss = train_contrastive_loss * c_loss_factor + train_gen_loss

        if USE_TEST_SET:
            test_contrastive_loss = 0
            test_gen_loss = 0
            for batch_idx, (data, labels) in enumerate(test_loader):
                data, labels = data.to(device), labels.to(device)
                offset = torch.randint(0, max_offset, (1,))
                if torch.sum(labels < eps) > 0:
                    data = data[:, offset:data.shape[1] - max_offset + offset]
#                    data_disturbed = data + torch.randn_like(data) * data_noise_amp
                    fakes = postprocess(net_g(preprocess(data.unsqueeze(1)))).squeeze(1)
                    fakes[data < eps] = 0

                    g_data = fakes[labels < eps]
                    g_labels = torch.ones((g_data.shape[0],), device=device)
                    g_data = g_data[:, (g_data.shape[1] + segment_size_disc) // 2 - segment_size_disc:(g_data.shape[1] + segment_size_disc) // 2]
                    outputs = net_d(preprocess(g_data.unsqueeze(1)))
                    loss = criterion(outputs, g_labels.unsqueeze(1))
                    test_gen_loss += loss.item()

                    loss = contrastive_loss(fakes, data, gaussian_filter_sigma)
                    test_contrastive_loss += loss.item()

            test_contrastive_loss /= len(test_loader)
            test_gen_loss /= len(test_loader)
            test_loss = test_contrastive_loss * c_loss_factor + test_gen_loss


        if epoch % 1 == 0:
            print(f"Epoch: {epoch:d}")
            print(f"t_loss: {train_loss:.6f} t_loss_c: {train_contrastive_loss:.6f} t_loss_g: {train_gen_loss:.6f} t_loss_d: {train_disc_loss:.8f}")
            if USE_TEST_SET:
                print(f"v_loss: {test_loss:.6f} v_loss_c: {test_contrastive_loss:.6f} v_loss_g: {test_gen_loss:.6f}")
            checkpoint = { 
                'epoch': epoch,
                'net_g': net_g.state_dict(),
                'net_d': net_d.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                'optimizer_d': optimizer_d.state_dict()}
            while True:
                try:
                    torch.save(net_g.state_dict(), MODEL_FILE) 
                    break
                except:
                    pass
            torch.save(checkpoint, TMP_FILE)
            if os.path.isfile(CHECKPOINT_FILE):
                while True:
                    try:
                        os.remove(CHECKPOINT_FILE)
                        break
                    except:
                        pass
            while True:
                try:
                    os.rename(TMP_FILE, CHECKPOINT_FILE)
                    break
                except:
                    pass
            try:
                #            np.save(FAKE_DATA_FILE, fakes)
                pass
            except:
                pass
            print(f"Data saved.")
        if True:#(USE_TEST_SET and test_loss < best_loss) or ((not USE_TEST_SET) and train_loss < best_loss):
            #            best_loss = test_loss if USE_TEST_SET else train_loss 
            BAK_FILE = name + " " + str(epoch // EPOCH_PER_BAK * EPOCH_PER_BAK) + ".pt"
            while True:
                try:
                    torch.save(net_g.state_dict(), BAK_FILE) 
                    break
                except:
                    pass
            print(f"Model backed up to '{BAK_FILE:s}'")


if __name__ == "__main__":
    random.seed(42)

    train_target_data, train_others_data, test_target_data, test_others_data = load_data()
    train_model("model", train_target_data, train_others_data, test_target_data, test_others_data)
