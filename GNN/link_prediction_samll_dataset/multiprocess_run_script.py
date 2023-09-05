import subprocess
import multiprocessing
import sys
from multiprocessing.pool import ThreadPool


def work(cmd):
    # output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    # output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    output = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    print(output.stdout)
    print(output.stderr)
    # output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, bufsize=1)
    return output


def main():
    count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=4)
    cmd = ["bash", "&&", "conda", "activate", "torch_gpu", "&&", "python", "seal_link_pred_for_small_data_with_dp.py"]
    for i in range(5):
        pool.apply_async(work, args=(cmd,))
    pool.close()
    pool.join()


def call_proc(cmd: list):
    """
    Args:
        cmd:list

    Returns:

    """
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, universal_newlines=True, bufsize=1, encoding='utf-8')
    while p.poll() is None:
        line = p.stdout.readline()
        line = line.strip()
        if line:
            print(f"subprogram {p.pid} output:\n{line}")
    if p.returncode == 0:
        print("subprogram success")
    else:
        print("subprogram failed")
    out, err = p.communicate()
    return (out, err)


def get_file_name(path):
    f_list = os.listdir(path)
    script_list = []
    for i in f_list:
        if os.path.splitext(i)[1] == ".sh" and os.path.basename(i)[0:8] == "running_":
            script_list.append(i)
            print(f"running {i}")
    return script_list


def generate_cmds(file_names):
    # cmd_list = [["bash", "&&", "conda", "activate", "torch_gpu", "&&", "sh", f"{file_name}"] for file_name in
    #             file_names]
    # cmd_list = [["D:\Git\git-bash.exe","&&", "conda", "activate", "torch_gpu","&&", "sh", f"{file_name}"] for file_name in
    #             file_names]
    # cmd_list = [["ping", "www.baidu.com"], ["ping", "www.baidu.com"]]
    # cmd_list = [["python", "seal_link_pred_for_small_data_with_dp.py"],
    #             ["python", "seal_link_pred_for_small_data_with_dp.py"]]
    cmd_list = [["sh", f"{file_name}"] for file_name in file_names]
    return cmd_list


if __name__ == '__main__':
    import os

    file_names = get_file_name("./")
    cmd_list = generate_cmds(file_names)
    # pool = multiprocessing.Pool(3)
    pool = ThreadPool(multiprocessing.cpu_count())
    results = []
    for cmd in cmd_list:
        results.append(pool.apply_async(call_proc, [cmd]))
    pool.close()
    pool.join()
    for result in results:
        out, err = result.get()
        print(f"out:{out}, err:{err}")

    # start_time = time.perf_counter()
    # # cmd = ["bash", "&&", "conda", "activate", "torch_gpu", "&&", "python", "seal_link_pred_for_small_data_with_dp.py"]
    # cmd = ["python", "seal_link_pred_for_small_data_with_dp.py"]
    # processes = [pool.apply_async(work, args=(cmd,)) for _ in range(5)]
    # result = [p.get() for p in processes]
    # finish_time = time.perf_counter()
    # print(f"Program finished in {finish_time - start_time} seconds")
    # print(result)
