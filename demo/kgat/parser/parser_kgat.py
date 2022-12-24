import argparse


def parse_kgat_args():
    parser = argparse.ArgumentParser(description="Run KGAT")

    parser.add_argument('--seed', type=int, default=2022)

    parser.add_argument('--data_name', nargs='?', default='amazon-book')
    parser.add_argument('--data_dir', nargs='?', default='datasets/')

    parser.add_argument('--use_pretrain', type=int, default=0)
    parser.add_argument('--pretrain_embedding_dir', nargs='?', default='datasets/pretrain/')
    parser.add_argument('--pretrain_model_path', nargs='?', default='trained_model/model.pth')

    parser.add_argument('--cf_batch_size', type=int, default=1024)
    parser.add_argument('--kg_batch_size', type=int, default=2048)
    parser.add_argument('--test_batch_size', type=int, default=10000)

    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--relation_dim', type=int, default=64)

    parser.add_argument('--laplacian_type', type=str, default='random-walk')
    parser.add_argument('--aggregation_type', type=str, default='bi-interaction')
    parser.add_argument('--conv_dim_list', nargs='?', default='[64, 32, 16]')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1, 0.1]')

    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5)
    parser.add_argument('--cf_l2loss_lambda', type=float, default=1e-5)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_epoch', type=int, default=1000)
    parser.add_argument('--stopping_steps', type=int, default=10)

    parser.add_argument('--cf_print_every', type=int, default=1)
    parser.add_argument('--kg_print_every', type=int, default=1)
    parser.add_argument('--evaluate_every', type=int, default=10)

    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]')

    args = parser.parse_args()

    save_dir = f'trained_model/KGAT/{args.data_name}/embed-dim{args.embed_dim}' \
               f'_relation-dim{args.relation_dim}_{args.laplacian_type}_{args.aggregation_type}' \
               f'_{"-".join([str(i) for i in eval(args.conv_dim_list)])}' \
               f'_lr{args.lr}_pretrain{args.use_pretrain}/'
    args.save_dir = save_dir

    return args
