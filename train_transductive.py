import copy
import logging
import os

from absl import app
from absl import flags
import torch
from torch.nn.functional import cosine_similarity
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from bgrl import *

log = logging.getLogger(__name__)
FLAGS = flags.FLAGS
flags.DEFINE_integer('model_seed', None, 'Random seed used for model initialization and training.')
flags.DEFINE_integer('data_seed', 1, 'Random seed used to generate train/val/test split.')
flags.DEFINE_integer('num_eval_splits', 3, 'Number of different train/test splits the model will be evaluated over.')

# Dataset.
flags.DEFINE_enum('dataset', 'coauthor-cs',
                  ['amazon-computers', 'amazon-photos', 'coauthor-cs', 'coauthor-physics', 'wiki-cs'],
                  'Which graph dataset to use.')
flags.DEFINE_string('dataset_dir', './data', 'Where the dataset resides.')

# Architecture.
flags.DEFINE_multi_integer('graph_encoder_layer', None, 'Conv layer sizes.')
flags.DEFINE_integer('predictor_hidden_size', 512, 'Hidden size of projector.')

# Training hyperparameters.
flags.DEFINE_integer('epochs', 10000, 'The number of training epochs.')
flags.DEFINE_float('lr', 1e-5, 'The learning rate for model training.')
flags.DEFINE_float('weight_decay', 1e-5, 'The value of the weight decay for training.')
flags.DEFINE_float('mm', 0.99, 'The momentum for moving average.')
flags.DEFINE_integer('lr_warmup_epochs', 1000, 'Warmup period for learning rate.')

# Augmentations.
flags.DEFINE_float('drop_edge_p_1', 0., 'Probability of edge dropout 1.')
flags.DEFINE_float('drop_feat_p_1', 0., 'Probability of node feature dropout 1.')
flags.DEFINE_float('drop_edge_p_2', 0., 'Probability of edge dropout 2.')
flags.DEFINE_float('drop_feat_p_2', 0., 'Probability of node feature dropout 2.')

# Logging and checkpoint.
flags.DEFINE_string('logdir', None, 'Where the checkpoint and logs are stored.')
flags.DEFINE_integer('log_steps', 10, 'Log information at every log_steps.')

# Evaluation
flags.DEFINE_integer('eval_epochs', 5, 'Evaluate every eval_epochs.')


def main(argv):
    # use CUDA_VISIBLE_DEVICES to select gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log.info('Using {} for training.'.format(device))

    # set random seed
    if FLAGS.model_seed is not None:
        log.info('Random seed set to {}.'.format(FLAGS.model_seed))
        set_random_seeds(random_seed=FLAGS.model_seed)

    # create log directory
    os.makedirs(FLAGS.logdir, exist_ok=True)
    with open(os.path.join(FLAGS.logdir, 'config.cfg'), "w") as file:
        file.write(FLAGS.flags_into_string())  # save config file

    # load data
    if FLAGS.dataset != 'wiki-cs':
        dataset = get_dataset(FLAGS.dataset_dir, FLAGS.dataset)
        num_eval_splits = FLAGS.num_eval_splits
    else:
        dataset, train_masks, val_masks, test_masks = get_wiki_cs(FLAGS.dataset_dir)
        num_eval_splits = train_masks.shape[1]

    data = dataset[0]  # all dataset include one graph
    log.info('Dataset {}, {}.'.format(dataset.__class__.__name__, data))
    data = data.to(device)  # permanently move in gpy memory

    # prepare transforms
    transform_1 = get_graph_drop_transform(drop_edge_p=FLAGS.drop_edge_p_1, drop_feat_p=FLAGS.drop_feat_p_1)
    transform_2 = get_graph_drop_transform(drop_edge_p=FLAGS.drop_edge_p_2, drop_feat_p=FLAGS.drop_feat_p_2)

    # build networks
    input_size, representation_size = data.x.size(1), FLAGS.graph_encoder_layer[-1]
    encoder = GCN([input_size] + FLAGS.graph_encoder_layer, batchnorm=True)   # 512, 256, 128
    predictor = MLP_Predictor(representation_size, representation_size, hidden_size=FLAGS.predictor_hidden_size)
    model = BGRL(encoder, predictor).to(device)

    # optimizer
    optimizer = AdamW(model.trainable_parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    # scheduler
    lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_epochs, FLAGS.epochs)
    mm_scheduler = CosineDecayScheduler(1 - FLAGS.mm, 0, FLAGS.epochs)

    # setup tensorboard and make custom layout
    writer = SummaryWriter(FLAGS.logdir)
    layout = {'accuracy': {'accuracy/test': ['Multiline', [f'accuracy/test_{i}' for i in range(num_eval_splits)]]}}
    writer.add_custom_scalars(layout)

    def train(step):
        model.train()

        # update learning rate
        lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # update momentum
        mm = 1 - mm_scheduler.get(step)

        # forward
        optimizer.zero_grad()

        x1, x2 = transform_1(data), transform_2(data)

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

    def eval(epoch):
        # make temporary copy of encoder
        tmp_encoder = copy.deepcopy(model.online_encoder).eval()
        representations, labels = compute_representations(tmp_encoder, dataset, device)

        if FLAGS.dataset != 'wiki-cs':
            scores = fit_logistic_regression(representations.cpu().numpy(), labels.cpu().numpy(),
                                             data_random_seed=FLAGS.data_seed, repeat=FLAGS.num_eval_splits)
        else:
            scores = fit_logistic_regression_preset_splits(representations.cpu().numpy(), labels.cpu().numpy(),
                                                           train_masks, val_masks, test_masks)

        for i, score in enumerate(scores):
            writer.add_scalar(f'accuracy/test_{i}', score, epoch)

    for epoch in tqdm(range(1, FLAGS.epochs + 1)):
        train(epoch-1)
        if epoch % FLAGS.eval_epochs == 0:
            eval(epoch)

    # save encoder weights
    torch.save({'model': model.online_encoder.state_dict()}, os.path.join(FLAGS.logdir, 'bgrl-wikics.pt'))


if __name__ == "__main__":
    log.info('PyTorch version: %s' % torch.__version__)
    app.run(main)
