import argparse


def parse_args():
    """
    设置模型的超参数
    """
    parser = argparse.ArgumentParser(description="Run KGAT")
    # 设置随机种子
    parser.add_argument('--seed', type=int, default=2022)
    # 设置数据集名称和路径
    parser.add_argument('--data_name', nargs='?', default='amazon-book')
    parser.add_argument('--data_dir', nargs='?', default='datasets/')
    # 设置batch size
    parser.add_argument('--cf_batch_size', type=int, default=1024)
    parser.add_argument('--kg_batch_size', type=int, default=2048)
    parser.add_argument('--test_batch_size', type=int, default=10000)
    # 设置embedding的维度
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--relation_dim', type=int, default=64)

    parser.add_argument('--laplacian_type', type=str, default='random-walk')
    parser.add_argument('--aggregation_type', type=str, default='bi-interaction')
    parser.add_argument('--conv_dim_list', nargs='?', default='[64, 32, 16]')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1, 0.1]')
    # 正则化系数
    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5)
    parser.add_argument('--cf_l2loss_lambda', type=float, default=1e-5)
    # 学习率, epoch数, early stopping数
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_epoch', type=int, default=1000)
    parser.add_argument('--stopping_steps', type=int, default=10)

    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]')

    args = parser.parse_args()

    save_dir = f'trained_model/KGAT/{args.data_name}/embed-dim{args.embed_dim}' \
               f'_relation-dim{args.relation_dim}_{args.laplacian_type}_{args.aggregation_type}' \
               f'_{"-".join([str(i) for i in eval(args.conv_dim_list)])}' \
               f'_lr{args.lr}/'
    args.save_dir = save_dir

    return args
