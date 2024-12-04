import io
import gzip
import os.path as osp
from tqdm import tqdm

"""
    step1 解压
    zip_data -> raw_data
"""

data_folder = "./zip_data"
out_foler = "./raw_data"


def extract_logs(filepath, hostid):
    search_pattern = f'SysClient{hostid}'
    output_filename = f'SysClient{hostid}.systemia.com.txt'

    with gzip.open(osp.join(data_folder, filepath), 'rt', encoding='utf-8') as fin:
        with open(osp.join(out_foler, output_filename), 'ab') as f:
            out = io.BufferedWriter(f)
            for line in fin:
                if search_pattern in line:
                    out.write(line.encode('utf-8'))
            out.flush()


def prepare_test_set():
    log_files = [
        ("AIA-201-225.ecar-2019-12-08T11-05-10.046.json.gz", "0201"),
        ("AIA-201-225.ecar-last.json.gz", "0201"),
        # ("AIA-501-525.ecar-2019-11-17T04-01-58.625.json.gz", "0501"),
        # ("AIA-501-525.ecar-last.json.gz", "0501"),
        # ("AIA-51-75.ecar-last.json.gz", "0051")
    ]

    # os.system("rm SysClient0201.com.txt")
    # os.system("rm SysClient0501.com.txt")
    # os.system("rm SysClient0051.com.txt")

    for file, code in tqdm(log_files, desc="Extracting logs", unit="file"):
        extract_logs(file, code)


if __name__ == "__main__":
    prepare_test_set()
