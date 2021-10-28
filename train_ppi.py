import copy
import logging
import os

from absl import app
from absl import flags
import torch
from torch.nn.functional import cosine_similarity
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.datasets import PPI
from tqdm import tqdm

from bgrl import *

log = logging.getLogger(__name__)
FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', None, 'Random seed.')
flags.DEFINE_integer('num_workers', 1, 'Number of CPU workers for dataloader.')

# Dataset.
flags.DEFINE_string('dataset_dir', './data', 'Where the dataset resides.')

# Architecture.
flags.DEFINE_integer('predictor_hidden_size', 4096, 'Hidden size of predictor.')

# Training hyperparameters.
flags.DEFINE_integer('steps', 10000, 'The number of training epochs.')
flags.DEFINE_integer('batch_size', 22, 'Number of graphs used in a batch.')
flags.DEFINE_float('lr', 0.02, 'The learning rate for model training.')
flags.DEFINE_float('weight_decay', 5e-4, 'The value of the weight decay.')
flags.DEFINE_float('mm', 0.99, 'The momentum for moving average.')
flags.DEFINE_integer('lr_warmup_steps', 1000, 'Warmup period for learning rate.')

# Augmentations.
flags.DEFINE_float('drop_edge_p_1', 0., 'Probability of edge dropout 1.')
flags.DEFINE_float('drop_feat_p_1', 0., 'Probability of node feature dropout 1.')
flags.DEFINE_float('drop_edge_p_2', 0., 'Probability of edge dropout 2.')
flags.DEFINE_float('drop_feat_p_2', 0., 'Probability of node feature dropout 2.')

# Logging and checkpoint.
flags.DEFINE_string('logdir', None, 'Where the checkpoint and logs are stored.')
flags.DEFINE_integer('log_steps', 10, 'Log information at every log_steps.')

# Evaluation
flags.DEFINE_integer('eval_steps', 2000, 'Evaluate every eval_epochs.')


def main(argv):
    # use CUDA_VISIBLE_DEVICES to select gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log.info('Using {} for training.'.format(device))

    # set random seed
    if FLAGS.seed is not None:
        log.info('Random seed set to {}.'.format(FLAGS.seed))
        set_random_seeds(random_seed=FLAGS.seed)

    # create log directory
    os.makedirs(FLAGS.logdir, exist_ok=True)
    with open(os.path.join(FLAGS.logdir, 'config.cfg'), "w") as file:
        file.write(FLAGS.flags_into_string())  # save config file

    # setup tensorboard
    writer = SummaryWriter(FLAGS.logdir)

    # load data
    train_dataset = PPI(FLAGS.dataset_dir, split='train')
    val_dataset = PPI(FLAGS.dataset_dir, split='val')
    test_dataset = PPI(FLAGS.dataset_dir, split='test')
    log.info('Dataset {}, graph 0: {}.'.format(train_dataset.__class__.__name__, train_dataset[0]))

    # train BGRL using both train and val splits
    train_loader = DataLoader(ConcatDataset([train_dataset, val_dataset]), batch_size=FLAGS.batch_size, shuffle=True,
                              num_workers=FLAGS.num_workers)

    # prepare transforms
    transform_1 = get_graph_drop_transform(drop_edge_p=FLAGS.drop_edge_p_1, drop_feat_p=FLAGS.drop_feat_p_1)
    transform_2 = get_graph_drop_transform(drop_edge_p=FLAGS.drop_edge_p_2, drop_feat_p=FLAGS.drop_feat_p_2)

    # build networks
    input_size, representation_size = train_dataset.num_node_features, 512
    encoder = GraphSAGE_GCN(input_size, 512, 512)
    predictor = MLP_Predictor(representation_size, representation_size, hidden_size=FLAGS.predictor_hidden_size)
    model = BGRL(encoder, predictor).to(device)

    # optimizer
    optimizer = AdamW(model.trainable_parameters(), lr=0., weight_decay=FLAGS.weight_decay)

    # scheduler
    lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_steps, FLAGS.steps)
    mm_scheduler = CosineDecayScheduler(1 - FLAGS.mm, 0, FLAGS.steps)

    def train(data, step):
        model.train()

        # move data to gpu and transform
        data = data.to(device)
        x1, x2 = transform_1(data), transform_2(data)

        # update learning rate
        lr = lr_scheduler.get(step)
        for g in optimizer.param_groups:
            g['lr'] = lr

        # update momentum
        mm = 1 - mm_scheduler.get(step)

        # forward
        optimizer.zero_grad()
        q1, y2 = model(x1, x2)
        q2, y1 = model(x2, x1)
        loss = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()

        loss.backward()
        # update online network
        optimizer.step()
        # update target network
        model.update_target_network(mm)

        # log scalars
        writer.add_scalar('params/lr', lr, step)
        writer.add_scalar('params/mm', mm, step)
        writer.add_scalar('train/loss', loss, step)


    def eval(step):
        tmp_encoder = copy.deepcopy(model.online_encoder).eval()

        train_data = compute_representations(tmp_encoder, train_dataset, device)
        val_data = compute_representations(tmp_encoder, val_dataset, device)
        test_data = compute_representations(tmp_encoder, test_dataset, device)

        val_f1, test_f1 = ppi_train_linear_layer(train_dataset.num_classes, train_data, val_data, test_data, device)
        writer.add_scalar('accuracy/val', val_f1, step)
        writer.add_scalar('accuracy/test', test_f1, step)

    train_iter = iter(train_loader)

    for step in tqdm(range(1, FLAGS.steps + 1)):
        data = next(train_iter, None)
        if data is None:
            train_iter = iter(train_loader)
            data = next(train_iter, None)

        train(data, step)

        if step % FLAGS.eval_steps == 0:
            eval(step)

    # save encoder weights
    torch.save({'model': model.online_encoder.state_dict()}, os.path.join(FLAGS.logdir, 'bgrl-wikics.pt'))


if __name__ == "__main__":
    log.info('PyTorch version: %s' % torch.__version__)
    app.run(main)
