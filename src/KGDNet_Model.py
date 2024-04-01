import tqdm
import torch
import torch.nn as nn
import torch.nn.init as init
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import RGCNConv, GCNConv, Linear

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LinConvolutionalModel(nn.Module):
  def __init__(self):
    super(LinConvolutionalModel, self).__init__()
    self.lin_conv = nn.Conv2d(1, 64, kernel_size=1)
    init.xavier_uniform_(self.lin_conv.weight)

    self.lin_pooling = nn.AdaptiveAvgPool2d(1)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.lin_conv(x.unsqueeze(1))
    x = self.relu(x)
    x = self.lin_pooling(x)
    x = x.view(x.size(0), -1)

    return x

################### GNNs ###################

# GNN for Clinical KGs
class RGCN(nn.Module):
  def __init__(self, vocab_size, input_channels, hidden_channels, out_channels, num_relations, dropout_rate=0.5):
    super(RGCN, self).__init__()

    self.embedding = nn.Embedding(vocab_size, input_channels)

    self.gnn1 = RGCNConv(input_channels, hidden_channels, num_relations=num_relations)
    init.xavier_uniform_(self.gnn1.weight)

    self.dropout = nn.Dropout(p=dropout_rate)

    self.gnn2 = RGCNConv(hidden_channels, out_channels, num_relations=num_relations)
    init.xavier_uniform_(self.gnn2.weight)

    self.relu = nn.ReLU()

    self.pooling = nn.AdaptiveAvgPool2d(out_channels)

  def forward(self, x, edge_index, edge_type):
    x = self.embedding(x)
    x = self.gnn1(x, edge_index, edge_type)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.gnn2(x, edge_index, edge_type)
    x = self.relu(x)
    x = self.pooling(x)

    return x
  
# GNN for Medicine KGs
class GCN(nn.Module):
  def __init__(self, vocab_size, input_channels, hidden_channels, out_channels, dropout_rate=0.5):
    super(GCN, self).__init__()

    self.embedding = nn.Embedding(vocab_size, input_channels)

    self.gnn1 = GCNConv(input_channels, hidden_channels, normalize=False)
    init.xavier_uniform_(self.gnn1.lin.weight)

    self.dropout = nn.Dropout(p=dropout_rate)

    self.gnn2 = GCNConv(hidden_channels, out_channels, normalize=False)
    init.xavier_uniform_(self.gnn2.lin.weight)

    self.relu = nn.ReLU()

    self.pooling = nn.AdaptiveAvgPool2d(out_channels)

  def forward(self, x, edge_index, edge_weight):
    x = self.embedding(x.squeeze(1).to(device))
    x = self.gnn1(x.to(device), edge_index.to(device), edge_weight.to(device))
    x = self.relu(x)
    x = self.dropout(x)
    x = self.gnn2(x, edge_index, edge_weight)
    x = self.relu(x)

    return x
  
#################### RNN ####################
class GRUNet(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.3):
    super(GRUNet, self).__init__()
    self.hidden_dim = hidden_dim
    self.n_layers = n_layers

    self.gru = nn.GRU(input_dim, hidden_dim, n_layers, dropout=drop_prob)
    for name, param in self.gru.named_parameters():
      if 'weight' in name:
        init.xavier_uniform_(param.data)

    self.lin = nn.Linear(hidden_dim, output_dim, bias=True)
    init.xavier_uniform_(self.lin.weight)

    self.relu = nn.ReLU()

  def forward(self, x, h):
    out, h = self.gru(x, h)
    out = self.relu(out)
    return out, h

  def init_hidden(self):
    weight = next(self.parameters()).data
    hidden = weight.new(self.n_layers, self.hidden_dim).zero_().to(device)
    return hidden
  
############### Fusion Module ###############

# Convolution Layer
class FusionConv(nn.Module):
  def __init__(self, input_channels, hidden_channels, out_channels, num_layers):
    super(FusionConv, self).__init__()

    self.initial_conv = nn.Conv1d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=1)
    init.xavier_uniform_(self.initial_conv.weight)

    self.intermediate_layers = nn.ModuleList([
        nn.Sequential(
            nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=1),
            nn.ReLU()
        )
        for _ in range(num_layers)
    ])
    for layer in self.intermediate_layers:
      for name, param in layer.named_parameters():
        if 'weight' in name:
          init.xavier_uniform_(param.data)

    self.final_conv = nn.Conv1d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1)
    init.xavier_uniform_(self.final_conv.weight)

    self.lin_conv = LinConvolutionalModel()

  def forward(self, x):
    x = self.initial_conv(x.unsqueeze(2))
    x = self.lin_conv(x)

    for layer in self.intermediate_layers:
      x = layer(x.unsqueeze(2))
      x = self.lin_conv(x)

    x = self.final_conv(x.unsqueeze(2))
    x = self.lin_conv(x)

    return x
  
# MLP Layer
class FusionMLP(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(FusionMLP, self).__init__()

    self.layer1 = nn.Linear(input_size, hidden_size, bias=True)
    init.xavier_uniform_(self.layer1.weight)

    self.layer2 = nn.Linear(hidden_size, output_size*2, bias=True)
    init.xavier_uniform_(self.layer2.weight)

    self.relu = nn.ReLU()

    self.conv_model = LinConvolutionalModel()

  def forward(self, x):
    x = self.layer1(x)
    x = self.relu(x)
    x = self.layer2(x)
    return x

################ Recommender ################
  
# Prediction MLP
class PredictionMLP(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super(PredictionMLP, self).__init__()
    self.layer1 = nn.Linear(input_dim, hidden_dim, bias=True)
    init.xavier_uniform_(self.layer1.weight)

    self.relu = nn.ReLU()

    self.layer2 = nn.Linear(hidden_dim, output_dim, bias=True)
    init.xavier_uniform_(self.layer2.weight)

    self.tanh = nn.Tanh()

  def forward(self, x):
    x = self.layer1(x)
    x = self.relu(x)
    x = self.layer2(x)
    x = self.tanh(x)
    return x
  
# Attention Module
class AttentionModule(nn.Module):
  def __init__(self, embed_dim, output_dim, num_heads, dropout=0.5):
    super(AttentionModule, self).__init__()

    self.MHA = nn.MultiheadAttention(embed_dim, num_heads, dropout)

    self.layer_norm = nn.LayerNorm(embed_dim)

    self.predictor = PredictionMLP(embed_dim, 2*embed_dim, output_dim)

  def forward(self, query, key, value, clinical_emb):
    attn_output, _ = self.MHA(query, key, value, need_weights=False)

    norm_output = self.layer_norm(clinical_emb + attn_output)

    pred_output = self.predictor(norm_output)

    return pred_output
  
############### KGDNet Model ###############
class KGDNet(nn.Module):
  def __init__(
      self,
      embed_dim, batch_size,
      clinical_vocab_size=3500, clinical_relations=12,
      medicine_vocab_size=145, medicine_relations=2, ddi_relations=1,
      emb_rnn_layers=3,
      fusion_layers=3,
      joint_rnn_layers=3,
      apm_heads=4
  ):
    super(KGDNet, self).__init__()

    self.num_clinical_nodes = clinical_vocab_size+1
    self.num_medicine_nodes = medicine_vocab_size+1

    self.batch_size = batch_size

    self.clinical_gnn = RGCN(self.num_clinical_nodes, embed_dim, embed_dim, embed_dim, clinical_relations)
    self.medicine_gnn = GCN(self.num_medicine_nodes, embed_dim, embed_dim, embed_dim, medicine_relations)
    self.ddi_gnn = GCN(self.num_medicine_nodes-1, embed_dim, embed_dim, embed_dim, ddi_relations)

    self.lin_conv = LinConvolutionalModel()

    self.emb_rnn = GRUNet(embed_dim, embed_dim, embed_dim, emb_rnn_layers)

    self.fusion_conv = FusionConv(embed_dim, embed_dim, embed_dim, fusion_layers)

    self.fusion_mlp = FusionMLP(embed_dim, embed_dim*2, embed_dim,)

    self.joint_rnn = GRUNet(embed_dim*2, embed_dim, embed_dim, joint_rnn_layers)

    self.prediction_layer = AttentionModule(embed_dim, self.num_medicine_nodes, apm_heads)

  def forward(self, medical_data, ddi_KG):
    num_adm = len(medical_data)

    ############# Clinical GNN #############
    clinical_embeddings = []

    for i in range(num_adm):
      clinical_data = medical_data[i][0]

      clinical_loader = NeighborLoader(
        clinical_data,
        num_neighbors=[20, 10],
        batch_size=self.batch_size,
        input_nodes=torch.arange(self.num_clinical_nodes),
        drop_last=True
      )

      clinical_emb = torch.zeros(self.num_clinical_nodes, self.embed_dim, self.embed_dim).to(device)

      for batch_data in tqdm.tqdm(clinical_loader):
        batch_data = batch_data.to(device)
        out = self.clinical_gnn(batch_data.x, batch_data.edge_index, batch_data.edge_type)
        for idx in range(len(out)):
          clinical_emb[batch_data.x[idx][0]] += out[idx]

      clinical_embeddings_agg = torch.mean(clinical_emb, dim=0, keepdim=True)
      clinical_embeddings_agg = self.lin_conv(clinical_embeddings_agg)
      clinical_embeddings.append(clinical_embeddings_agg)

    ############## Medicine RNN ##############
    medicine_embeddings = []

    for i in range(num_adm):
      medicine_data = medical_data[i][1]
      medicine_data = T.AddSelfLoops(attr = 'edge_weights', fill_value = 1)(medicine_data)
      medicine_out = self.medicine_gnn(medicine_data.x, medicine_data.edge_index, medicine_data.edge_weights)

      ddi_out = self.ddi_gnn(ddi_KG.x.to(device), ddi_KG.edge_index.to(device), torch.ones(1055).to(device))
      ddi_out = torch.cat((ddi_out, nn.Dropout(p=0.5)(torch.rand(1, 64)).to(device)))

      medicine_out = medicine_out - ddi_out

      medicine_embeddings_agg = torch.mean(medicine_out, dim=0, keepdim=True)
      medicine_embeddings.append(medicine_embeddings_agg)

    ############# RNN Embeddings #############
    clinical_hidden_feats = self.emb_rnn.init_hidden().to(device)
    clinical_rnn_feats = []
    for emb in clinical_embeddings:
      _, clinical_hidden_feats = self.emb_rnn(emb.to(device), clinical_hidden_feats.data.to(device))
      clinical_hidden_feats_agg = torch.sum(clinical_hidden_feats, dim=0, keepdim=True)
      clinical_rnn_feats.append(clinical_hidden_feats_agg)

    medicine_hidden_feats = self.emb_rnn.init_hidden().to(device)
    medicine_rnn_feats = []
    for emb in medicine_embeddings:
      _, medicine_hidden_feats = self.emb_rnn(emb, medicine_hidden_feats.data)
      medicine_hidden_feats_agg = torch.sum(medicine_hidden_feats, dim=0, keepdim=True)
      medicine_rnn_feats.append(medicine_hidden_feats_agg)

    ################# Fusion #################
    concat_feats = []

    for i in range(num_adm):
      concat_feats.append(torch.cat((clinical_rnn_feats[i], medicine_rnn_feats[i])))

    for i in range(num_adm):
      concat_feats[i] = self.fusion_conv(concat_feats[i])
      concat_feats[i] = torch.sum(concat_feats[i], dim=0, keepdim=True)

    joint_feats = []
    for i in range(num_adm):
      joint_feats.append(self.fusion_mlp(concat_feats[i]))

    ################ Joint RNN ################
    joint_hidden_feats = self.joint_rnn.init_hidden().to(device)

    joint_rnn_feats = []
    joint_rnn_feats.append(torch.zeros(1, self.embed_dim))

    for feat in joint_feats:
      _, joint_hidden_feats = self.joint_rnn(feat, joint_hidden_feats.data)
      joint_hidden_feats_agg = torch.sum(joint_hidden_feats, dim=0, keepdim=True)
      joint_rnn_feats.append(joint_hidden_feats_agg)

    ########### Medicine Predictions ###########
    med_out = [None] * num_adm

    for i in range(num_adm):
      med_out[i] = self.prediction_layer(
          clinical_rnn_feats[i].to(device),
          joint_rnn_feats[i].to(device),
          joint_rnn_feats[i].to(device),
          clinical_embeddings[i].to(device)
      )

    return med_out