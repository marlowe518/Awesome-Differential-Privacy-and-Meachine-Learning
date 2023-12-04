from seal_link_pred_for_small_data_with_dp import account_privacy

if __name__ == "__main__":
    args.neighborhood_subgraph = True
    orders = np.arange(1, 10, 0.1)[1:]
    num_message_passing_steps = 1000000
    max_node_degree = 60
    num_hops = 1
    step_num = 50
    batch_size = 512
    train_num = 100000
    sigma = 0.09
    temp, _ = account_privacy(num_message_passing_steps,
                              max_node_degree,
                              num_hops, step_num,
                              batch_size, train_num, sigma,
                              orders=orders)
    print(f"privacy budget:{temp}")
