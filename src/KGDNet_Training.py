import pickle
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys
sys.path.append("..")

from KGDNet_Model import KGDNet
from util import ddi_score, eval, test
from dataloader import get_ehr_data

resume_path = ''

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', default=False, help="test mode")
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--embed_dim', type=int, default=64, help='embedding dimension size')
parser.add_argument('--ddi_thresh', type=float, default=0.08, help="Set DDI threshold")
parser.add_argument('--resume_path', type=str, default=resume_path, help='resume path')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
  ehr_KGs, patient_info, ddi_KG, ddi_adj = get_ehr_data(device)

  split_point = int(len(ehr_KGs) * 2 / 3)
  eval_len = int(len(ehr_KGs[split_point:]) / 2)

  ehr_train = ehr_KGs[:split_point]
  ehr_test = ehr_KGs[split_point:split_point + eval_len]
  ehr_eval = ehr_KGs[split_point+eval_len:]

  patient_info_train = patient_info[:split_point]
  patient_info_test = patient_info[split_point:split_point + eval_len]
  patient_info_eval = patient_info[split_point+eval_len:]
  
  ## Training ##
  metrics_arr = []

  model = KGDNet(embed_dim=args.embed_dim, batch_size=args.batch_size).to(device)

  if args.test:
    model.load_state_dict(torch.load(open(args.resume_path, 'rb')))
    test(model, ehr_test, patient_info_test, ddi_KG, ddi_adj, args.ddi_thresh, device)
    return

  optimizer = optim.RMSprop(model.parameters(), lr=2e-4, weight_decay=1e-4, alpha=0.995)

  num_epochs = 100

  for epoch in range(num_epochs):
    model.train()

    epoch_loss = 0
    epoch_roc_auc = 0
    epoch_prauc = 0
    epoch_f1 = 0
    epoch_jaccard = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_meds = 0
    epoch_ddi = 0

    num_patients = len(ehr_train)

    for subj_id in range(num_patients):
      patient_data = ehr_train[subj_id]
      num_adm = len(patient_data)

      ## Get ground truths
      ground_truths = []

      ddi_gt = torch.zeros(146).to(device)

      for i in range(num_adm):
        gt = torch.zeros(146).to(device)

        try:
          meds = patient_info_train[subj_id][i][('patient', 'prescribed_to', 'medicine')]['edges'][1]
          gt[meds] = 1
        except: gt[145] = 1

        ddi_gt += gt

        ground_truths.append(gt)

      ##############################
      # Zero the gradients
      optimizer.zero_grad()

      outputs = model(patient_data, ddi_KG)
      ##############################

      ## Apply threshold ##
      output_thresh = [np.zeros(146)] * num_adm
      for i in range(num_adm):
        for j in range(146):
          output_thresh[i][j] = (outputs[i][0][j] > -0.95)
        output_thresh[i] = output_thresh[i].astype(int)

      loss = 0
      bce_loss = 0
      mlm_loss = 0
      ddi_loss = 0

      ## Recommendation Loss ##
      for i in range(num_adm):
        bce_loss += F.binary_cross_entropy_with_logits(outputs[i][0], ground_truths[i])
        mlm_loss += F.multilabel_margin_loss(outputs[i][0], ground_truths[i].long())
      bce_loss = bce_loss / num_adm
      mlm_loss = mlm_loss / num_adm

      ddi_loss_param, ddi_loss = ddi_score(outputs, output_thresh, ddi_gt, ddi_adj, args.ddi_thresh)

      ## Total Loss ##
      rec_loss_param = 0.95
      loss = ddi_loss_param * (
          rec_loss_param * bce_loss + (1 - rec_loss_param) * mlm_loss
      ) + (1 - ddi_loss_param) * ddi_loss

      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
      optimizer.step()

      epoch_loss += loss
      epoch_ddi += ddi_loss

    epoch_loss = loss / num_patients

    epoch_roc_auc, epoch_prauc, epoch_f1, 
    epoch_jaccard, epoch_precision, 
    epoch_recall, epoch_meds, epoch_ddi = eval(
      model, ehr_eval, patient_info_eval, ddi_KG, ddi_adj, args.ddi_thresh, device
    )

    metrics_arr.append([epoch+1, epoch_loss.item(), epoch_roc_auc, epoch_prauc, epoch_f1, epoch_jaccard, epoch_precision, epoch_recall, epoch_ddi, epoch_meds])

    print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, ValAUC: {epoch_roc_auc:.4f}, PRAUC: {epoch_prauc:.4f}, F1: {epoch_f1:.4f}, Jaccard: {epoch_jaccard:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, DDI: {epoch_ddi:.4f}, num_meds: {epoch_meds:.4f}")

  with open('metrics.pkl', 'wb') as f:
    pickle.dump(metrics_arr, f)