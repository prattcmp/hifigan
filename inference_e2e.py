from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from .env import AttrDict
from .meldataset import MAX_WAV_VALUE
from .models import Generator

h = None
device = None

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(output_file, checkpoint_file, input_mel=None, input_mel_file=None):
    assert input_mel is not None or input_mel_file is not None, "An input mel (file or npy) must be provided"


    config_file = os.path.join(os.path.split(checkpoint_file)[0], 'config_v2.json')
    with open(config_file) as f:
        data = f.read()


    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if input_mel is not None:
        device = input_mel.device
    elif torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        if input_mel is not None:
            # Load mel from numpy variable
            x = input_mel
        else:
            # Load mel from filepath
            x = np.load(os.path.join(a.input_mel_file, filname))
            x = torch.FloatTensor(x).to(device)
        y_g_hat = generator(x)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')

        write(output_file + ".wav", h.sampling_rate, audio)
        print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mel_file', default='test_mel_file')
    parser.add_argument('--output_file', default='generated_file_from_mel')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    inference(a.output_file, a.checkpoint_file, input_mel_file=a.input_mel_file)


if __name__ == '__main__':
    main()

