import subprocess
import multiprocessing


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


if __name__ == '__main__':
    import time

    pool = multiprocessing.Pool(3)
    start_time = time.perf_counter()
    # cmd = ["bash", "&&", "conda", "activate", "torch_gpu", "&&", "python", "seal_link_pred_for_small_data_with_dp.py"]
    cmd = ["python", "seal_link_pred_for_small_data_with_dp.py"]
    processes = [pool.apply_async(work, args=(cmd,)) for _ in range(5)]
    result = [p.get() for p in processes]
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time - start_time} seconds")
    print(result)
