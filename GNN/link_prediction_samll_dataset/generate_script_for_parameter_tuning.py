import time
import math


def generate_script():
    dataset = ["Celegans"]
    with open(f"./running_{'&'.join(dataset)}_{time.strftime('%Y-%m-%d-%H-%M-%S')}.sh", "w") as f:
        # for dataset in ["Yeast", "Ecoli", "Power", "Router"]:
        # for dataset in ["NS"]:
        # for dataset in ["USAir"]:
        for dataset in dataset:
            # for dataset in ["NS"]:
            for num_hops in [2]:
                for max_node_degree in [40]:
                    # for lr in [0.001, 0.01, 0.1, 1.]:
                    for lr in [0.01]:
                        # for sigma in [0.2, 2.]:
                        for target_epsilon in [0.2, 0.4, 0.8, 0.93, 0.95, 0.97]:
                            # for target_epsilon in [1, 10, 20]:
                            for sigma in [1.]:
                                for max_norm in [1.]:
                                    for batch_size in [128]:
                                        # TODO num_layers should be deduced by number of hops
                                        general_args = ["CUDA_VISIBLE_DEVICES=0",
                                                        "python", "seal_link_pred_for_small_data_with_dp.py",
                                                        f"--data_name {dataset}",
                                                        # f"--num_hops {num_hops}",
                                                        f"--num_layers {num_hops}",
                                                        # f"--max_node_degree {max_node_degree}",
                                                        f"--lr {lr}",
                                                        f"--sigma {sigma}",
                                                        f"--max_norm {max_norm}",
                                                        f"--batch_size {batch_size}",
                                                        f"--target_epsilon {target_epsilon}",
                                                        f"--runs {5}",
                                                        f"--epochs {30}",
                                                        # f"--dp_method {'DPLP'}",
                                                        # f"--neighborhood_subgraph",
                                                        f"--uniq_appendix '20230905'"
                                                        ]
                                        for dp_method in ["DPLP", "LapGraph", "DPGC", "DPLP-NS"]:
                                            if dp_method != "DPLP":  # The opposite cases are based on neighborhood subgraph
                                                ns_hop = math.ceil((num_hops - 1) / 2)
                                                args = general_args + [f"--neighborhood_subgraph"]
                                                if dp_method == "DPGC" or dp_method == "DPLP-NS":  # These two methods are in-process perturbation based on DPLP
                                                    if dp_method == "DPGC":
                                                        args += [f"--max_node_degree {300}"]
                                                    else:
                                                        args += [f"--max_node_degree {max_node_degree}"]
                                                    dp_method = "DPLP"
                                                args += [f"--num_hops {ns_hop}", f"--dp_method '{dp_method}'"]
                                            else:
                                                args = general_args + [f"--max_node_degree {max_node_degree}",
                                                                       f"--num_hops {num_hops}",
                                                                       f"--dp_method '{dp_method}'"]
                                            print(" ".join(args), file=f)


if __name__ == "__main__":
    generate_script()
