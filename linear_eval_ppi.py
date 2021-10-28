import logging

from absl import app
from absl import flags
import torch
from torch_geometric.datasets import PPI

from bgrl import *

log = logging.getLogger(__name__)
FLAGS = flags.FLAGS

# Dataset.
flags.DEFINE_string('dataset_dir', './data', 'Where the dataset resides.')
flags.DEFINE_string('ckpt_path', None, 'Path to checkpoint.')


def main(argv):
    # use CUDA_VISIBLE_DEVICES to select gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log.info('Using {} for evaluation.'.format(device))

    # load data
    train_dataset = PPI(FLAGS.dataset_dir, split='train')
    val_dataset = PPI(FLAGS.dataset_dir, split='val')
    test_dataset = PPI(FLAGS.dataset_dir, split='test')
    log.info('Dataset {}, graph 0: {}.'.format(train_dataset.__class__.__name__, train_dataset[0]))

    # build networks
    input_size, representation_size = train_dataset.num_node_features, 512
    encoder = GraphSAGE_GCN(input_size, 512, 512)
    load_trained_encoder(encoder, FLAGS.ckpt_path, device)
    encoder.eval()

    # compute representations
    train_data = compute_representations(encoder, train_dataset, device)
    val_data = compute_representations(encoder, val_dataset, device)
    test_data = compute_representations(encoder, test_dataset, device)

    val_f1, test_f1 = ppi_train_linear_layer(train_dataset.num_classes, train_data, val_data, test_data, device)
    print('Test F1-score: %.5f' % test_f1)


if __name__ == "__main__":
    log.info('PyTorch version: %s' % torch.__version__)
    app.run(main)
