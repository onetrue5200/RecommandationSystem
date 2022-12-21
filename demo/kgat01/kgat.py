import torch
import torch.nn as nn
import torch.nn.functional as F


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class Aggregator(nn.Module):
    
    def __init__(self, in_dim, out_dim, dropout, agg_type):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.agg_type = agg_type

        self.message_dropout = nn.Dropout(self.dropout)
        self.activation = nn.LeakyReLU()

        if self.agg_type == "gcn":
            pass
        elif self.agg_type == "graphsage":
            pass
        elif self.agg_type == "bi-interaction":
            self.linear1 = nn.Linear(self.in_dim, self.out_dim)
            self.linear2 = nn.Linear(self.in_dim, self.out_dim)
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)
        else:
            raise NotImplementedError
    
    def forward(self, ego_embeddings, A_in):
        side_embeddings = torch.matmul(A_in, ego_embeddings)

        if self.agg_type == "gcn":
            pass
        elif self.agg_type == "graphsage":
            pass
        elif self.agg_type == "bi-interaction":
            sum_embeddings = self.activation(self.linear1(ego_embeddings + side_embeddings))
            bi_embeddings = self.activation(self.linear2(ego_embeddings * side_embeddings))
            embeddings = bi_embeddings + sum_embeddings
        else:
            raise NotImplementedError
        
        embeddings = self.message_dropout(embeddings)
        return embeddings


class KGAT(nn.Module):
    
    def __init__(self, args, data):
        super().__init__()
        self.args = args
        self.data = data

        self.users_entities_embedding = nn.Embedding(self.data.n_users_entities, self.args.embed_dim)
        self.relations_embedding = nn.Embedding(self.data.n_relations, self.args.relation_dim)
        self.trans_M = nn.Parameter(torch.Tensor(self.data.n_relations, self.args.embed_dim, self.args.relation_dim))

        nn.init.xavier_uniform_(self.users_entities_embedding.weight)
        nn.init.xavier_uniform_(self.relations_embedding.weight)
        nn.init.xavier_uniform_(self.trans_M)

        self.conv_dim_list = [self.args.embed_dim] + eval(self.args.conv_dim_list)
        self.mess_dropout = eval(self.args.mess_dropout)
        self.n_layers = len(eval(self.args.conv_dim_list))
        
        self.A_in = nn.Parameter(self.data.A_in, requires_grad=False)

        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(Aggregator(
                                                    self.conv_dim_list[k],
                                                    self.conv_dim_list[k + 1],
                                                    self.mess_dropout[k],
                                                    self.args.aggregation_type
                                                )
                                            )
    
    def calc_cf_embeddings(self):
        """
        使用embedding计算3层agg后的embedding
        结果cat起来返回
        """
        ego_embedding = self.users_entities_embedding.weight
        all_embedding = [ego_embedding]
        for agg in self.aggregator_layers:
            ego_embedding = agg(ego_embedding, self.A_in)
            norm_embedding = F.normalize(ego_embedding, p=2, dim=1)
            all_embedding.append(norm_embedding)
        
        return torch.cat(all_embedding, dim=1)

    def calc_cf_loss(self, users, pos_items, neg_items):
        all_embedding = self.calc_cf_embeddings()
        
        users_embedding = all_embedding[users]
        pos_items_embedding = all_embedding[pos_items]
        neg_items_embedding = all_embedding[neg_items]

        pos_score = torch.sum(users_embedding * pos_items_embedding, dim=1)
        neg_score = torch.sum(users_embedding * neg_items_embedding, dim=1)

        cf_loss = torch.mean((-1.0) * F.logsigmoid(pos_score - neg_score))
        
        l2_loss = _L2_loss_mean(users_embedding) + _L2_loss_mean(pos_items_embedding) + _L2_loss_mean(neg_items_embedding)
        loss = cf_loss + self.args.cf_l2loss_lambda * l2_loss
        return loss

    def forward(self, *input, mode):
        if mode == "train_cf":
            return self.calc_cf_loss(*input)

