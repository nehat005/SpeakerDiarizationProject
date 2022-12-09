import os
import glob
import numpy as np


def write_scp_file(path_to_rttms: str, scpf_file_path: str):
    """
    Write <utterance id> <corresponding rttm file path> into a 'scp' file format

    :param path_to_rttms: Path to directory containing rttm files
    :param scpf_file_path: Path to scpf file to write the data
    :return: None
    """
    rttm_files = glob.glob(path_to_rttms + '/*.rttm')

    with open(scpf_file_path, 'w') as out_file:
        for file in rttm_files:
            filename = os.path.basename(file).split('.')[0]
            out_file.write('{} {}\n'.format(filename, file))


def read_rttm_file(file_path: str) -> np.array:
    """
    Read rttm data line-wise

    :param file_path: Path to rttm file to be read
    :return: numpy array file data of rttm file stored row-wise
    """

    rttm_file = file_path
    with open(rttm_file, 'r') as rttm_file_id:
        data = rttm_file_id.readlines()
        data_lines = []
        for i in range(len(data)):
            data_lines.append(data[i].split())

    return np.vstack(data_lines)
