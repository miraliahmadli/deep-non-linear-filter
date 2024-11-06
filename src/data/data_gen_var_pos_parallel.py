"""
Dataset generation for a target speaker in a variable location.
The target speaker angle is relative to the microphone array orientation.
This preprocessing script creates a HDF5 file with three datasets:
- train
- val
- test

Each dataset has the shape [NUM_SAMPLES, 3, CHANNELS, MAX_SAMPLES_PER_FILE].
In the second axis, we store in this order the
- spatialized target signal (includes reverb)
- the spatialized noise signal (sum of all interfering speakers)
- the dry target signal (including the time-shift caused by the direct path)


The code in this file was partially written by Nils Mohrmann.
"""

import os
import glob
import json
import multiprocessing as mp
import random
from typing import List

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
import soundfile as sf


random.seed(12345)

# WSJ0 dataset path
WSJ0_PATH = "/informatik2/sp/intern/databases_DEPRECATED/Good/WSJ/CSR-1-WSJ-0/WAV/wsj0"
# Path where to save the simulated data
SIM_DATA_PATH = "prep/conditional/"


class RoomSimulation:

    def __init__(self, num_channels):
        """
        Room simulation of sp acoustic lab with 3 dimensions

        :param channels: number of microphones in a uniform circular array
        """
        self.channels = num_channels
        self.room_dim = np.array([9.3, 5.04, 2.84])
        self.rt60_tgt = 0.3  # only a init value
        e_absorption, max_order = pra.inverse_sabine(
            self.rt60_tgt, self.room_dim)

        self.room = pra.ShoeBox(self.room_dim,
                                fs=16000,
                                materials=pra.Material(e_absorption),
                                max_order=max_order)

    def set_room_properties(self, rt: float, room_dim: np.ndarray):
        """
        Recreate room with a new reverberation time, this deletes all sources and mics.
        :param rt: reverberation time
        :param room_dim: room dimension ([x,y,z])
        :return: None
        """
        self.rt60_tgt = rt
        if self.rt60_tgt > 0:
            e_absorption, max_order = pra.inverse_sabine(
                self.rt60_tgt, room_dim)

            self.room = pra.ShoeBox(room_dim,
                                    fs=16000,
                                    materials=pra.Material(e_absorption),
                                    max_order=max_order)
        else:
            e_absorption, max_order = pra.inverse_sabine(0.5, room_dim)
            self.room = pra.ShoeBox(room_dim,
                                    fs=16000,
                                    materials=pra.Material(e_absorption),
                                    max_order=0)

    def reset(self):
        """
        Reset the room, delete all sources and microphones
        Keeps RT60
        :return: None
        """
        self.set_room_properties(self.rt60_tgt, self.room_dim)

    def set_microphones(
        self,
        x: float,
        y: float,
        z: float,
        phi: float,
        mic_offset: np.ndarray,
    ):
        """
        Add microphone array at position xyz with rotation phi
        Radius: 0.05 m
        :param x: x pos
        :param y: y pos
        :param z: z pos
        :param phi: The counterclockwise rotation of the first element in the array (from the x-axis)
        :return:
        """
        if self.channels == 2:
            # special orientation for 2 mics. -> speaker at broadside
            phi += np.pi / 2
        R = pra.beamforming.circular_2D_array(
            [x, y], self.channels, phi0=phi, radius=0.05)
        R = np.vstack((R, [[z] * self.channels]))
        R += mic_offset
        self.room.add_microphone_array(pra.Beamformer(R, self.room.fs))

    def add_source(self, position: np.ndarray, signal: np.ndarray, delay: float):
        """
        Add signal source in room with a delay
        :param position: position [x, y, z]
        :param signal: The signal played by the source
        :param delay: A time delay until the source signal starts in the simulation
        :return: None
        """
        self.room.add_source(position, signal, delay)  # type: ignore

    def plot(self):
        """
        Plot the room (see pyroomacoustic examples)
        :return: None
        """
        fig, ax = self.room.plot()
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 6])
        ax.set_zlim([0, 3])
        ax.view_init(elev=90, azim=0)

        # Now compute the delay and sum weights for the beamformer
        # room.mic_array.rake_delay_and_sum_weights(room.sources[0][:1])
        # plot the room and resulting beamformer
        # room.plot(freq=[1000, 2000, 4000, 8000], img_order=0)
        plt.show()

        # self.room.compute_rir()
        # room.plot_rir()
        # plt.plot(self.room.rir[1][0])
        # plt.show()

    def measuretime(self):
        """
        Get measured RT60
        :return: rt60 in seconds
        """
        self.room.compute_rir()
        return self.room.measure_rt60()


class SPRoomSimulator:
    """
    Generate dataset for the training of the steerable non-linear filter.
    """

    def __init__(self, num_channels=3, seed=13, mode="train"):
        self.training = True
        if mode == "train":
            path = "si_tr_s"
        elif mode == "val":
            path = "si_dt_20"
        elif mode == "test":
            path = "si_et_05"
            self.training = False
        else:
            raise ValueError("Mode not supported")
        path = f'{WSJ0_PATH}/{path}'

        self.speaker = glob.glob(path + "/*")

        rng = np.random.default_rng(seed)
        rng.shuffle(self.speaker)
        self.channels = num_channels
        self.exp_room = RoomSimulation(num_channels=self.channels)
        self.dry_room = RoomSimulation(num_channels=self.channels)
        self.fs = 16000

    def create_sample(
        self,
        speaker_list: List[str],
        seed2: int,
        target_angle: float = 0,
        reverb: bool = True,
        rt60_min: float = 0.2,
        rt60_max: float = 1,
        snr_min: int = -10,
        snr_max: int = 5,
        min_dist=0.8,
        max_dist=1.2,
        mic_pert_std: float = 0,
        min_angle_dist: int = 10,
        target_idx: int = 0,
        return_dict_idx: int = 0,
        return_dict: dict = {},
    ):
        """
        Create for a list of speech signals (first one is the target signal) the spatial image using a randomly placed
        microphone array and distributing the interfering speakers (len(speaker_list)-1) uniformly around the array.

        :param speaker_list: List of paths to speaker utterances
        :param seed2: Seed for the random audio files and positions
        :param target_angle: The DOA of the target speaker in degree.
        :param reverb: Create reverberant signals
        :param rt60_min, rt60_mx: The RT60 is sampled uniformly from the range (rt60_min, rt60_max)
        :param snr: The SNR is sampled uniformly from the range (snr_min, snr_max). The noise signal is rescaled to match the chosen SNR. If snr_min is None, no rescaling is performed.
        :param min_dist, max_dist: The range (min_dist, max_dist) from which the sources (also interfering sources) are sampled uniformly. Unit is meters.
        :param mic_pert_std: Add noise to the microphone positions sampled from a Gaussian with zero mean and specified standard deviation. Unit is cm.
        :param min_angle_dist: Minimum angle distance between two sources (target-interfering and interfering-interfering)

        :return: the audio signals as numpy array [N_SPEAKERS, N_CHANNELS, N_SAMPLES] and corresponding meta data
        """
        # set seed for this sample
        rng = np.random.default_rng(seed2)
        meta = {}

        exp_room = RoomSimulation(num_channels=self.channels)
        dry_room = RoomSimulation(num_channels=self.channels)

        signal = []
        for file in speaker_list:
            audio, fs = sf.read(file)
            signal.append(audio / np.max(np.abs(audio)) * 0.3)

        # ensure noise signal is long enough and does not start with zeros always
        offset_indices = np.random.randint(
            low=-8000, high=8000, size=len(speaker_list)-1)
        target_signal_len = len(signal[0])
        for i in range(len(speaker_list)-1):
            new_signal = np.roll(
                np.resize(signal[1+i], target_signal_len), shift=offset_indices[i])
            signal[1+i] = new_signal

        # room configuration
        RT = rng.uniform(rt60_min, rt60_max) if reverb else 0
        meta["rt"] = RT

        room_dim = np.squeeze(np.array([rng.uniform(
            2.5, 5, 1), rng.uniform(3, 9, 1), rng.uniform(2.2, 3.5, 1)]))
        meta["room_dim"] = [room_dim[0], room_dim[1], room_dim[2]]

        exp_room.set_room_properties(RT, np.array(room_dim))
        dry_room.set_room_properties(0, np.array(room_dim))

        # mic array at random position in room (min 1.2 m to wall)
        mic_pos = rng.random(3) * (room_dim - 2.42) + 1.21
        mic_pos[2] = 1.5

        if mic_pert_std > 0:
            mic_offset = rng.normal(
                loc=0, scale=mic_pert_std, size=(3, self.channels))
        else:
            mic_offset = np.zeros((3, self.channels))

        phi = rng.random() * 2 * np.pi  # microphone rotation
        exp_room.set_microphones(
            mic_pos[0], mic_pos[1], mic_pos[2], phi, mic_offset)
        dry_room.set_microphones(
            mic_pos[0], mic_pos[1], mic_pos[2], phi, mic_offset)

        meta["mic_pos"] = mic_pos.tolist()
        meta["mic_phi"] = phi

        # target speaker
        target_phi = phi + target_angle / 360 * 2 * np.pi
        speaker_phis = [target_phi]
        main_source = mic_pos + \
            normal_vec(target_phi) * \
            ((rng.random() * (max_dist-min_dist) + min_dist))
        main_source[2] = rng.normal(1.60, 0.08)  # height of speaker

        exp_room.add_source(main_source, signal[0], 0)
        dry_room.add_source(main_source, signal[0], 0)

        meta["target_file"] = speaker_list[0].split(
            "wsj0")[-1].replace("\\", "/")
        meta["n_samples"] = len(signal[0])
        meta["target_pos"] = main_source.tolist()
        meta["target_angle"] = target_angle
        n_interfering = len(speaker_list) - 1
        for interf_idx, interf_path in enumerate(speaker_list[1:]):

            # distance max 1.2 m, min 0.8 m
            min_angle_dist_rad = 2*np.pi/360*min_angle_dist
            speaker_range = (2*np.pi-2*min_angle_dist_rad)/n_interfering

            too_close = True
            while too_close:  # make sure the selected angle is not too close to other source
                speaker_phi = target_phi + min_angle_dist_rad + speaker_range * \
                    rng.random() + interf_idx * speaker_range
                interf_source = mic_pos + \
                    normal_vec(speaker_phi) * (rng.random()
                                               * (max_dist-min_dist) + min_dist)

                # height of speaker is round about the height of standing people
                interf_source[2] = rng.normal(1.60, 0.08)

                if len(speaker_phis) == 0:
                    too_close = False
                    speaker_phis.append(speaker_phi)
                else:
                    if speaker_phi - speaker_phis[-1] < np.deg2rad(min_angle_dist) or\
                            (speaker_phis[0] + 2*np.pi) - speaker_phi < np.deg2rad(min_angle_dist):
                        # previous speaker or first speaker too close
                        too_close = True
                    else:
                        too_close = False
                        speaker_phis.append(speaker_phi)

            exp_room.add_source(interf_source, signal[interf_idx + 1], 0)
            meta[f"interf{interf_idx}_file"] = interf_path.split(
                "wsj0")[-1].replace("\\", "/")
            meta[f"interf{interf_idx}_pos"] = interf_source.tolist()

        # return_premix allows separation of speaker signals
        exp_room.room.compute_rir()
        mic_signals = exp_room.room.simulate(return_premix=True)

        # direct path target
        dry_room.room.compute_rir()
        target_signal = dry_room.room.simulate(return_premix=True)

        # scale to SNR
        reverb_target_signal = mic_signals[0, ...]
        noise_signal = np.sum(mic_signals[1:, ...], axis=0)
        target_signal = target_signal[0, ...]

        if not snr_min is np.nan:
            target_snr = rng.uniform(snr_min, snr_max)
            noise_factor = snr_scale_factor(
                reverb_target_signal, noise_signal, target_snr)
            noise_signal = noise_signal * noise_factor

            meta["snr"] = target_snr

        return_dict[return_dict_idx] = (
            reverb_target_signal, noise_signal, target_signal, meta, target_idx)
        return reverb_target_signal, noise_signal, target_signal, meta, target_idx

    def create_samples(
        self,
        speaker_lists: List[List[str]],
        seeds: List[int],
        target_angles: List[float] = [0.0],
        reverb: bool = True,
        rt60_min: float = 0.2,
        rt60_max: float = 1,
        snr_min: int = -10,
        snr_max: int = 5,
        mic_pert_std: float = 0,
        min_dist=0.8,
        max_dist=1.2,
        min_angle_dist: int = 10,
        target_indices: List[int] = []
    ):
        '''Use all cpus to create samples in parallel'''
        processes = []
        manager = mp.Manager()
        return_dict = manager.dict()
        for i, (speaker_list, seed, target_angle, target_idx) in\
                enumerate(zip(speaker_lists, seeds, target_angles, target_indices)):
            p = mp.Process(
                target=self.create_sample,
                args=(
                    speaker_list,
                    seed,
                    target_angle,
                    reverb,
                    rt60_min,
                    rt60_max,
                    snr_min,
                    snr_max,
                    min_dist,
                    max_dist,
                    mic_pert_std,
                    min_angle_dist,
                    target_idx,
                    i,
                    return_dict,
                ))
            processes.append(p)
            p.start()  # start the process

        # wait for all processes to finish
        for p in processes:
            p.join()
        return [return_dict[i] for i in range(len(speaker_lists))]

    def get_room(self):
        return self.exp_room.room

    def plot(self):
        self.exp_room.plot()


def normal_vec(phi):
    return np.array([np.cos(phi), np.sin(phi), 0])


def snr_scale_factor(speech: np.ndarray, noise: np.ndarray, snr: int):
    """
    Compute the scale factor that has to be applied to a noise signal in order for the noisy (sum of noise and clean)
    to have the specified SNR.

    :param speech: the clean speech signal [..., SAMPLES]
    :param noise: the noise signal [..., SAMPLES]
    :param snr: the SNR of the mixture
    :return: the scaling factor
    """

    noise_var = np.mean(np.var(noise, axis=-1))
    speech_var = np.mean(np.var(speech, axis=-1))

    factor = np.sqrt(
        speech_var / np.maximum((noise_var * 10. ** (snr / 10.)), 10**(-6)))

    return factor


def prep_speaker_mix_data(
    store_dir: str,
    post_fix: str | None = None,
    wsj0_path: str = 'whatever',
    n_channels: int = 3,
    n_interfering_speakers: int = 3,
    target_fs: int = 16000,
    num_files: dict = {'train': -1,
                       'val': -1,
                       'test': -1},
    angle_settings: dict | None = None,
    reverb: bool = True,
    side_room: int = 10,
    rt60_min=0.2,
    rt60_max=0.8,
    snr_min=-10,
    snr_max=5,
    mic_pert=0,
    min_dist=0.8,
    max_dist=1.2,
    batch_size=300,
):
    """
    Preparation of speaker mix dataset. The target speaker is placed in a fixed position relative to the microphone
    array. The interfering speakers are placed randomly with one speaker per angle segment.

    If angle_settings are provided, the function can also create a dataset with a moving speaker placed
    on a range of angles.

    :param store_dir: path to directory in which to store the dataset
    :param post_fix: postfix to specify the characteristics of the dataset
    :param wsj0_path: path the the raw WSJ0 data
    :param n_channels: number of channels in the microphone array
    :param n_interfering_speakers: the number of interfering speakers
    :param target_fs: the target sampling rate for the dataset
    :param num_files: a dictionary specifying the number of examples per stage
    :param angle_settings: a dict {'start': -45, 'stop': 45, 'step': 1, 'n_samples_per_angle': 100}
    :param reverb: turn off reverberation if set to False
    :param rt60_min: min RT60 time (uniformly sampled if reverb)
    :param rt60_max: max RT60 time (uniformly sampled if reverb)
    :param snr_min: min SNR (uniformly sampled)
    :param snr_max: max SNR (uniformely sampled)
    :param side_room: minimum angle difference between two sources (default: 10 deg)

    """
    assert angle_settings is not None, "Please provide angle settings"

    prep_store_name = f"prep_mix{'_' + post_fix if post_fix else ''}.hdf5"

    train_samples = list(
        sorted(glob.glob(os.path.join(wsj0_path, 'si_tr_s/*/*.wav'))))
    val_samples = list(
        sorted(glob.glob(os.path.join(wsj0_path, 'si_dt_20/*/*.wav'))))
    test_samples = list(
        sorted(glob.glob(os.path.join(wsj0_path, 'si_et_05/*/*.wav'))))

    n_angles = len(range(angle_settings['start'],
                         angle_settings['stop'], angle_settings['step']))

    meta = {}
    with h5py.File(os.path.join(store_dir, prep_store_name), 'w') as prep_storage:
        for data_set, samples in (('train', train_samples),
                                  ('val', val_samples),
                                  ('test', test_samples)):
            if num_files[data_set] == 0:
                continue
            n_dataset_samples = num_files[data_set] if num_files[data_set] > 0 else len(
                samples)

            # Variable target speaker position distributed over some range
            n_dataset_samples_full = n_dataset_samples*n_angles
            angle_start = angle_settings['start']
            angle_stop = angle_settings['stop']
            angle_step = angle_settings['step']

            MAX_SAMPLES_PER_FILE = 12 * target_fs
            audio_dataset = prep_storage.create_dataset(data_set,
                                                        shape=(
                                                            n_dataset_samples_full, 3, n_channels, MAX_SAMPLES_PER_FILE),
                                                        chunks=(
                                                            1, 3, n_channels, MAX_SAMPLES_PER_FILE),
                                                        dtype=np.float32,
                                                        compression="gzip",
                                                        shuffle=True)

            set_meta = {}

            sproom = SPRoomSimulator(num_channels=n_channels, mode=data_set)

            speaker_lists = []
            seeds = []
            angles = []
            indices = []
            target_indices = []

            for i, fixed_angle in enumerate(range(angle_start, angle_stop, angle_step)):
                random.shuffle(samples)  # pick random speakers
                for target_idx, target_path in enumerate(samples[:n_dataset_samples]):
                    interfering_speakers = random.choices(
                        samples, k=n_interfering_speakers)
                    speaker_lists.append([target_path] + interfering_speakers)
                    seeds.append(i*n_dataset_samples+target_idx)
                    angles.append(fixed_angle)
                    target_indices.append(target_idx)
                    indices.append(i*n_dataset_samples+target_idx)

            for i in range(0, len(speaker_lists), batch_size):
                print("Processing batch", i)
                results = sproom.create_samples(
                    speaker_lists[i: i+batch_size],
                    seeds[i: i+batch_size],
                    target_angles=angles[i: i+batch_size],
                    reverb=reverb,
                    rt60_min=rt60_min,
                    rt60_max=rt60_max,
                    snr_min=snr_min,
                    snr_max=snr_max,
                    mic_pert_std=mic_pert,
                    min_dist=min_dist,
                    max_dist=max_dist,
                    min_angle_dist=side_room,
                    target_indices=target_indices[i: i+batch_size],
                )
                print("Batch done")
                for j, (reverb_target_signal, noise_signal, dry_target_signal, sample_meta, target_idx) in enumerate(results):
                    val = indices[i+j]
                    n_audio_samples = min(
                        sample_meta['n_samples'], MAX_SAMPLES_PER_FILE)
                    sample_meta['n_samples'] = n_audio_samples
                    sample_meta['target_dir'] = fixed_angle
                    set_meta[val] = sample_meta

                    # store reverb clean
                    audio_dataset[val, 0, :,
                                  :n_audio_samples] = reverb_target_signal[:, :n_audio_samples]

                    # store noise
                    audio_dataset[val, 1, :,
                                  :n_audio_samples] = noise_signal[:, :n_audio_samples]

                    # store dry clean
                    audio_dataset[val, 2, :,
                                  :n_audio_samples] = dry_target_signal[:, :n_audio_samples]

                    audio_dataset[val, :, :,
                                  n_audio_samples:MAX_SAMPLES_PER_FILE] = 0
                print("Batch stored")

            meta[data_set] = set_meta
    with open(
        os.path.join(store_dir,
                     f"prep_mix_meta{'_' + post_fix if post_fix else ''}.json"),
        'w', encoding='utf-8'
    ) as prep_meta_storage:
        json.dump(meta, prep_meta_storage, indent=4)


if __name__ == '__main__':
    BATCH_SIZE = 300  # works w/ RAM size of 32GB
    CHANNELS = 3
    prefix = f'ch{CHANNELS}_sp5_var_target'

    store_path = os.path.join(SIM_DATA_PATH)
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    prep_speaker_mix_data(
        store_path,
        prefix,
        WSJ0_PATH,
        n_interfering_speakers=5,
        n_channels=CHANNELS,
        num_files={'train': 300, 'val': 15,
                   'test': 10},  # files per angle!
        angle_settings={'start': -180,
                        'stop': 180, 'step': 2},
        reverb=True,
        rt60_min=0.2,
        rt60_max=0.5,
        snr_min=np.nan,  # type: ignore
        snr_max=np.nan,  # type: ignore
        mic_pert=0,
        min_dist=0.8,
        max_dist=1.2,
        side_room=10,
        batch_size=BATCH_SIZE,
    )
