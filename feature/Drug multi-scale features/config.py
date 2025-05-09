from argparse import ArgumentParser

def make_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument("--epochs",
                        type=int,
                        default=200,
                        help="number of epochs")
    parser.add_argument('--subgraph_dim',
                        type=int,
                        default=256,
                        help='dimension of subgraph features')
    parser.add_argument('--feature_dim',
                        type=int,
                        default=51,
                        help='dimension of atom features')
    parser.add_argument('--save_model',
                        type=int,
                        default=True,
                        help='whether save the model')
    parser.add_argument('--lr',
                        type=int,
                        default=1e-3,
                        help='learning rate of the optimizer')
    parser.add_argument('--lr_decay',
                        type=int,
                        default=0.75,
                        help='decay rate of the optimizer')
    parser.add_argument('--decay_interval',
                        type=int,
                        default=20,
                        help='number of decay rounds')
    parser.add_argument('--MolCLR_dim',
                        type=int,
                        default=512,
                        help='feature dimension of MolCLR output')

    args = parser.parse_args()

    return args
