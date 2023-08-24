import time


def generate_script():
    with open(f"./running_{time.strftime('%Y-%m-%d-%H-%M-%S')}.sh", "w") as f:
        # for dataset in ["Yeast", "Ecoli", "Power", "Router"]:
        # for dataset in ["NS"]:
        # for dataset in ["USAir"]:
        # for dataset in ["NS", "USAir", "PB", "Ecoli", "Power"]:
        for dataset in ["Power"]:
            for num_hops in [2, 3, 4, 5, 6, 7]:
                for max_node_degree in [40]:
                    # for lr in [0.001, 0.01, 0.1, 1.]:
                    for lr in [0.1]:
                        # for sigma in [0.2, 2.]:
                        for target_epsilon in [4]:
                            # for target_epsilon in [1, 10, 20]:
                            for sigma in [1.]:
                                for max_norm in [1.]:
                                    for batch_size in [128]:
                                        # TODO num_layers should be deduced by number of hops
                                        print(" ".join(["python", "seal_link_pred_for_small_data_with_dp.py",
                                                        f"--data_name {dataset}",
                                                        f"--num_hops {num_hops}",
                                                        f"--num_layers {num_hops}",
                                                        f"--max_node_degree {max_node_degree}",
                                                        f"--lr {lr}",
                                                        f"--sigma {sigma}",
                                                        f"--max_norm {max_norm}",
                                                        f"--batch_size {batch_size}",
                                                        f"--target_epsilon {target_epsilon}",
                                                        f"--runs {1}",
                                                        f"--dp_method {'DPLP'}",
                                                        # f"--neighborhood_subgraph",
                                                        f"--uniq_appendix '20230815'"]), file=f, )


# def split_script():
#     import multiprocessing
#     import numpy as np
#     cpu_counts = multiprocessing.cpu_count() - 1
#     with open("./running.sh", "r") as f:
#         lines = f.readlines()
#         # bins = len(lines) // cpu_counts
#         # remain_lines = len(lines) - bins
#         start_indexes = np.linspace(0, len(lines), cpu_counts, dtype=int)
#         for idx in range(len(start_indexes) - 1):
#             lines[idx:]
#         print(start_indexes)


if __name__ == "__main__":
    generate_script()
