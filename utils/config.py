import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch implementation of Classification models.')
    parser.add_argument('--model_name', type=str, default='vgg16', choices=['vgg16', 'resnet32', 'resnet56'])
    parser.add_argument('--model_num', type=int, default=2)
    parser.add_argument('--port_num', type=str, default='8097')
    parser.add_argument('--save_dir', type=str, default="./save/")
    parser.add_argument('--gamma', type=float, default=0.1, help='Value of learning rate decay')
    parser.add_argument('--is_train', type=bool, default=True)
    parser.add_argument('--dataroot', type=str, default='/home/nkim/data', help='path to dataset')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['mnist', 'cifar10', 'cifar100'])
    parser.add_argument('--save_load', type=bool, default=False)
    parser.add_argument('--save_location', type=str, default=False)
    parser.add_argument('--download', type=str, default='True')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='How num using process')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--k_number', type=str, default=[])
    parser.add_argument('--total_models', type=str, default=[])
    parser.add_argument('--lr_patience', type=int, default=10,
                        help='Number of epochs to wait before reducing lr')
    parser.add_argument('--best_model_num', type=int, default=1)
    parser.add_argument('--logs_dir', type=str, default="./logs")
    parser.add_argument('--use_tensorboard', type=bool, default=True,
                        help='Whether to use tensorboard for visualization')

    return parser.parse_args()
