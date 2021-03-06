import argparse



def get_parser():
    parser = argparse.ArgumentParser()

    # -- Data params ---
    parser.add_argument("--penalty_function", type=str, default="element_wise",
                        choices=['element_wise', 'group_lasso', 'perturbed_node'])
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--samplePerStep", type=int, required=True)
    parser.add_argument("--leftstep", type=int, required=True)
    parser.add_argument("--rightstep", type=int, required=True)
    parser.add_argument("--save_data_path", type=str, default="./data/")
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=8)

    return parser
