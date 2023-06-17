with open("./running.sh", "w") as f:
    for dataset in ["NS", "Ecoli", "Power", "Router", "Yeast"]:
        for max_node_degree in [1, 2, 3]:
            for num_hops in [1, 2, 3]:
                for lr in [0.1, 0.01, 0.001]:
                    for sigma in [0.2, 2.]:
                        print(" ".join(["python","seal_link_pred_for_small_data_with_dp.py",
                                        f"--data_name {dataset}",
                                        f"--num_hops {num_hops}",
                                        f"--num_layers {num_hops}",
                                        f"--max_node_degree {max_node_degree}",
                                        f"--lr {lr}",
                                        f"--sigma {sigma}"]), file=f)
