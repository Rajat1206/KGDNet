import numpy as np
import random
import pickle
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, jaccard_score, confusion_matrix, precision_score, recall_score

def ddi_score(outputs, output_thresh, ddi_gt, ddi_adj, ddi_thresh):
  num_adm = len(outputs)

  ## Patient DDI Calc. ##
  med_pair_ct = 0
  ddi_pair_ct = 0
  for i in range(146):
    for j in range(i+1, 146):
      if ddi_gt[i] != 0 and ddi_gt[j] != 0:
        med_pair_ct += 1
        ddi_pair_ct += int(ddi_adj[i][j])
  try:
    patient_ddi_rate = ddi_pair_ct / med_pair_ct
  except: patient_ddi_rate = 0

  ddi_loss_param = 1 if patient_ddi_rate <= ddi_thresh else max(
    0, 1-((patient_ddi_rate - ddi_thresh)/0.05)
  )

  ## DDI Rate Calc. ##
  ddi_rates = [0] * num_adm

  for adm in range(num_adm):
    med_pair_ct = 0
    ddi_pair_ct = 0
    for i in range(len(output_thresh[adm])):
      for j in range(i+1, len(output_thresh[adm])):
        if output_thresh[adm][i] == 1 and output_thresh[adm][j] == 1:
          med_pair_ct += 1
          ddi_pair_ct += int(ddi_adj[i][j])
    try: ddi_rates[adm] = ddi_pair_ct / med_pair_ct
    except: ddi_rates[adm] = 0
  ddi_loss = np.mean(ddi_rates)

  return ddi_loss_param, ddi_loss

def eval(model, ehr_eval, patient_info_eval, ddi_KG, ddi_adj, ddi_thresh, device):
  model.eval()

  epoch_roc_auc = 0
  epoch_prauc = 0
  epoch_f1 = 0
  epoch_jaccard = 0
  epoch_precision = 0
  epoch_recall = 0
  epoch_meds = 0
  epoch_ddi = 0

  num_patients = len(ehr_eval)

  for subj_id in range(num_patients):
    roc_auc = 0
    prauc = 0
    f1 = 0
    jaccard = 0
    num_meds = 0
    precision = 0
    recall = 0

    patient_data = ehr_eval[subj_id]
    num_adm = len(patient_data)

    ## Get ground truths
    ground_truths = []

    ddi_gt = torch.zeros(146).to(device)

    for i in range(num_adm):
      gt = torch.zeros(146).to(device)

      meds = patient_info_eval[subj_id][i][('patient', 'prescribed_to', 'medicine')]['edges'][1]
      gt[meds] = 1

      ddi_gt += gt

      ground_truths.append(gt)

    outputs = model(patient_data, ddi_KG)

    output_thresh = [np.zeros(146)] * num_adm
    for i in range(num_adm):
      for j in range(146):
        output_thresh[i][j] = (outputs[i][0][j] > ddi_thresh)
      output_thresh[i] = output_thresh[i].astype(int)

    for i in range(num_adm):
      output_numpy = outputs[i][0].cpu().detach().numpy()
      ground_truth_numpy = ground_truths[i].cpu().detach().numpy()
      weights_numpy = (ground_truth_numpy * 1.75) + 1

      roc_auc += roc_auc_score(ground_truth_numpy, output_numpy, sample_weight=weights_numpy)
      prauc += average_precision_score(ground_truth_numpy, output_numpy, sample_weight=weights_numpy)
      f1 += f1_score(ground_truth_numpy, output_thresh[i], sample_weight=weights_numpy, zero_division=0.0)
      jaccard += jaccard_score(ground_truth_numpy, output_thresh[i], sample_weight=weights_numpy, zero_division=0.0)

      precision += precision_score(ground_truth_numpy, output_thresh[i], sample_weight=weights_numpy, zero_division=0.0)
      recall += recall_score(ground_truth_numpy, output_thresh[i], sample_weight=weights_numpy, zero_division=0.0)

      cm = confusion_matrix(ground_truth_numpy, output_thresh[i])
      num_meds += cm[0][1] + cm[1][1]

    _, ddi_loss = ddi_score(outputs, output_thresh, ddi_gt, ddi_adj, ddi_thresh)

    roc_auc = roc_auc / num_adm
    prauc = prauc / num_adm
    f1 = f1 / num_adm
    jaccard = jaccard / num_adm
    precision = precision / num_adm
    recall = recall / num_adm
    num_meds = num_meds / num_adm

    epoch_roc_auc += roc_auc
    epoch_prauc += prauc
    epoch_f1 += f1
    epoch_jaccard += jaccard
    epoch_precision += precision
    epoch_recall += recall
    epoch_meds += num_meds
    epoch_ddi += ddi_loss

  epoch_roc_auc = epoch_roc_auc / num_patients
  epoch_prauc = epoch_prauc / num_patients
  epoch_f1 = epoch_f1 / num_patients
  epoch_jaccard = epoch_jaccard / num_patients
  epoch_precision = epoch_precision / num_patients
  epoch_recall = epoch_recall / num_patients
  epoch_meds = epoch_meds / num_patients
  epoch_ddi = epoch_ddi / num_patients

  return epoch_roc_auc, epoch_prauc, epoch_f1, epoch_jaccard, epoch_precision, epoch_recall, epoch_meds

def test(model, ehr_test, patient_info_test, ddi_KG, ddi_adj, ddi_thresh, device):
  metrics = []

  data_len = len(ehr_test)
  req_len = int(0.8 * data_len)

  for _ in 10:
    idx_list = list(range(data_len))
    random.shuffle(idx_list)
    idx_list = idx_list[:req_len]

    iter_ehr = ehr_test[idx_list]
    iter_patient_info = patient_info_test[idx_list]

    roc_auc, prauc, f1, jaccard, precision, recall, meds, ddi = eval(
      model, iter_ehr, iter_patient_info, ddi_KG, ddi_adj, ddi_thresh, device
    )

    print(f"ValAUC: {roc_auc:.4f}, PRAUC: {prauc:.4f}, F1: {f1:.4f}, Jaccard: {jaccard:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, DDI: {ddi:.4f}, num_meds: {meds:.4f}")
    metrics.append([roc_auc, prauc, f1, jaccard, precision, recall, meds, ddi])

  with open('test_metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

  return