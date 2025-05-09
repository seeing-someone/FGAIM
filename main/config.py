is_seq_in_graph = True
is_con_in_graph = True
is_profile_in_graph = True #改成False
is_emb_in_graph = True #改成False
NUM_EPOCHS = 2000#2000
TRAIN_BATCH_SIZE = 128 # 128
TEST_BATCH_SIZE = 256 # 256
run_model = 0
cuda = 0
setting = 0
# LR = 0.0005
LR = 0.0001

dataset = 'DrugAI'

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
