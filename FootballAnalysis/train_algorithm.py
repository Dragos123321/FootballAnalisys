import tqdm
import argparse
import pickle
import numpy as np
import os
import time

import torch
import torch.optim as optim

from FootAndBall.network import footandball
from FootAndBall.data.data_reader import make_dataloaders
from FootAndBall.network.ssd_loss import SSDLoss
from FootAndBall.misc.config import Params

MODEL_FOLDER = 'models'


def train_model_util(model, optimizer, scheduler, num_epochs, data_loaders, device, model_name):
    """
    Trains the model for the specified number of epochs using the provided data loaders.

    Args:
        model: The model to be trained.
        optimizer: The optimizer for updating the model's parameters.
        scheduler: The learning rate scheduler.
        num_epochs: The number of epochs to train the model.
        data_loaders: A dictionary containing the data loaders for the training and validation sets.
        device: The device to be used for training.
        model_name: The name of the model.

    Returns:
        training_stats: A dictionary containing the training statistics.
    """

    criterion = SSDLoss(neg_pos_ratio=3)
    training_stats = {'train': [], 'val': []}

    phases = get_training_phases(data_loaders)

    alpha_localization_player, alpha_confidence_player, alpha_confidence_ball = calculate_loss_weights()

    print('Training...')
    for _ in tqdm.tqdm(range(num_epochs)):
        for phase in phases:
            model.train() if phase == 'train' else model.eval()

            batch_stats = {'loss': [], 'loss_ball_confidence': [], 'loss_player_confidence': [],
                           'loss_player_localization': []}

            count_batches = 0
            for ndx, (images, boxes, labels) in enumerate(data_loaders[phase]):
                images, ground_truth_maps = prepare_data(images, boxes, labels, device, model)
                count_batches += 1

                with torch.set_grad_enabled(phase == 'train'):
                    loss, loss_confidence_ball, loss_confidence_player, loss_localization_player = \
                        calculate_loss(model,
                                       criterion,
                                       optimizer,
                                       images,
                                       ground_truth_maps,
                                       phase,
                                       alpha_localization_player,
                                       alpha_confidence_player,
                                       alpha_confidence_ball)

                count_batches += 1
                batch_stats = update_batch_stats(batch_stats, loss, loss_confidence_ball, loss_confidence_player,
                                                 loss_localization_player)

            avg_batch_stats = calculate_avg_batch_stats(batch_stats)
            training_stats[phase].append(avg_batch_stats)
            print_training_stats(phase, avg_batch_stats)

        scheduler.step()
        print('')

    save_model(model, model_name)
    save_training_stats(training_stats, model_name)

    return training_stats


def get_training_phases(data_loaders):
    if 'val' in data_loaders:
        return ['train', 'val']
    else:
        return ['train']


def calculate_loss_weights():
    alpha_localization_player = 0.01
    alpha_confidence_player = 1.0
    alpha_confidence_ball = 5.0

    total = alpha_localization_player + alpha_confidence_player + alpha_confidence_ball
    alpha_localization_player /= total
    alpha_confidence_player /= total
    alpha_confidence_ball /= total

    return alpha_localization_player, alpha_confidence_player, alpha_confidence_ball


def prepare_data(images, boxes, labels, device, model):
    images = images.to(device)
    height, width = images.shape[-2], images.shape[-1]
    ground_truth_maps = model.groundtruth_maps(boxes, labels, (height, width))
    ground_truth_maps = [m.to(device) for m in ground_truth_maps]

    return images, ground_truth_maps


def calculate_loss(model, criterion, optimizer, images, ground_truth_maps, phase,
                   alpha_localization_player, alpha_confidence_player, alpha_confidence_ball):
    predictions = model(images)
    optimizer.zero_grad()
    loss_localization_player, loss_confidence_player, loss_confidence_ball = criterion(predictions, ground_truth_maps)

    loss = alpha_localization_player * loss_localization_player + alpha_confidence_player * \
           loss_confidence_player + alpha_confidence_ball * loss_confidence_ball

    if phase == 'train':
        loss.backward()
        optimizer.step()

    return loss, loss_confidence_ball, loss_confidence_player, loss_localization_player


def update_batch_stats(batch_stats, loss, loss_confidence_ball, loss_confidence_player, loss_localization_player):
    batch_stats['loss'].append(loss.item())
    batch_stats['loss_ball_confidence'].append(loss_confidence_ball.item())
    batch_stats['loss_player_confidence'].append(loss_confidence_player.item())
    batch_stats['loss_player_localization'].append(loss_localization_player.item())

    return batch_stats


def calculate_avg_batch_stats(batch_stats):
    avg_batch_stats = {}
    for stat in batch_stats:
        avg_batch_stats[stat] = np.mean(batch_stats[stat])

    return avg_batch_stats


def print_training_stats(phase, avg_batch_stats):
    s = '{} Avg. loss total / ball confidence / player confidence / player localization: '
    '{:.4f} / {:.4f} / {:.4f} / {:.4f}'
    print(s.format(phase, avg_batch_stats['loss'], avg_batch_stats['loss_ball_confidence'],
                   avg_batch_stats['loss_player_confidence'], avg_batch_stats['loss_player_localization']))


def save_model(model, model_name):
    model_filepath = os.path.join(MODEL_FOLDER, model_name + '_final' + '.pth')
    torch.save(model.state_dict(), model_filepath)


def save_training_stats(training_stats, model_name):
    with open('training_stats_{}.pickle'.format(model_name), 'wb') as handle:
        pickle.dump(training_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)


def train_model(params: Params):
    # Create the model folder if it doesn't exist
    if not os.path.exists(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)

    # Check if the model folder exists
    assert os.path.exists(MODEL_FOLDER), 'Cannot create folder to save trained model: {}'.format(MODEL_FOLDER)

    # Load data loaders
    data_loaders = make_dataloaders(params)

    # Print dataset sizes
    print('Training set: Dataset size: {}'.format(len(data_loaders['train'].dataset)))
    if 'val' in data_loaders:
        print('Validation set: Dataset size: {}'.format(len(data_loaders['val'].dataset)))

    # Create model
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = footandball.model_factory(params.model, 'train')
    model.print_summary(show_architecture=True)
    model = model.to(device)

    # Generate model name based on current time
    model_name = 'model_' + time.strftime("%Y%m%d_%height%M")
    print('Model name: {}'.format(model_name))

    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    scheduler_milestones = [int(params.epochs * 0.75)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, scheduler_milestones, gamma=0.1)

    # Train the model
    train_model_util(model, optimizer, scheduler, params.epochs, data_loaders, device, model_name)


if __name__ == '__main__':
    print('Train FootAndBall model')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to the configuration file', type=str, default='config.txt')
    parser.add_argument('--debug', dest='debug', help='debug mode', action='store_true')
    args = parser.parse_args()

    print('Config path: {}'.format(args.config))
    print('Debug mode: {}'.format(args.debug))

    params = Params(args.config)
    params.print()

    train_model(params)
