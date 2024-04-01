import pickle
import torch
from torch_geometric.data import Data

def get_ehr_data(device):
  with open('ehr_atc3.pkl', 'rb') as f:
    ehr_records = pickle.load(f)

  ## Dict Maps
  with open('DictMaps/diag_map_dict.pkl', 'rb') as f:
    diag_map_dict = pickle.load(f)

  with open('DictMaps/diag_map_rev_dict.pkl', 'rb') as f:
    diag_map_rev_dict = pickle.load(f)

  with open('DictMaps/proc_map_dict.pkl', 'rb') as f:
    proc_map_dict = pickle.load(f)

  with open('DictMaps/proc_map_rev_dict.pkl', 'rb') as f:
    proc_map_rev_dict = pickle.load(f)

  with open('DictMaps/proc_map_adjusted_dict.pkl', 'rb') as f:
    proc_map_adjusted_dict = pickle.load(f)

  with open('DictMaps/proc_map_rev_adjusted_dict.pkl', 'rb') as f:
    proc_map_rev_adjusted_dict = pickle.load(f)

  with open('DictMaps/atc_map_dict.pkl', 'rb') as f:
    atc_map_dict = pickle.load(f)

  with open('DictMaps/atc_map_rev_dict.pkl', 'rb') as f:
    atc_map_rev_dict = pickle.load(f)

  with open('DictMaps/atc3_map_dict.pkl', 'rb') as f:
    atc3_map_dict = pickle.load(f)

  with open('DictMaps/atc3_map_rev_dict.pkl', 'rb') as f:
    atc3_map_rev_dict = pickle.load(f)

  ## Adjacency Matrices
  with open('AdjMatrices/diag_adj.pkl', 'rb') as f:
    diag_adj = pickle.load(f)

  with open('AdjMatrices/proc_adj.pkl', 'rb') as f:
    proc_adj = pickle.load(f)

  with open('AdjMatrices/diag_proc_adj.pkl', 'rb') as f:
    diag_proc_adj = pickle.load(f)

  with open('AdjMatrices/proc_diag_adj.pkl', 'rb') as f:
    proc_diag_adj = pickle.load(f)

  with open('AdjMatrices/prescriptions_adj.pkl', 'rb') as f:
    prescriptions_adj = pickle.load(f)

  with open('AdjMatrices/ddi_adj.pkl', 'rb') as f:
    ddi_adj = pickle.load(f)

  with open('AdjMatrices/prescriptions_atc3_adj.pkl', 'rb') as f:
    prescriptions_atc3_adj = pickle.load(f)

  with open('AdjMatrices/ddi_atc3_adj.pkl', 'rb') as f:
    ddi_atc3_adj = pickle.load(f)

  num_diag_nodes = len(diag_map_dict)
  num_proc_nodes = len(proc_map_dict)
  num_clinical_nodes = num_diag_nodes + num_proc_nodes
  num_med_nodes = len(atc3_map_dict)

  ## Clinical Data
  diag_diag_edges = torch.from_numpy(diag_adj).nonzero().t().contiguous().to(device)

  proc_proc_edges = torch.from_numpy(proc_adj).nonzero().t().contiguous().to(device)
  proc_proc_edges = torch.add(proc_proc_edges, num_diag_nodes)

  diag_proc_edges = torch.from_numpy(diag_proc_adj).nonzero().t().contiguous().to(device)
  diag_proc_edges[1] = torch.add(diag_proc_edges[1], num_diag_nodes)

  proc_diag_edges = torch.from_numpy(proc_diag_adj).nonzero().t().contiguous().to(device)
  proc_diag_edges[0] = torch.add(proc_diag_edges[0], num_diag_nodes)

  clinical_edges = {
      ('diagnoses', 'identified_with', 'diagnoses'): {
        "code": 0,
        "edges": diag_diag_edges,
        "edge_id": torch.tensor([0] * diag_diag_edges.size()[1]).to(device)
      },
      ('procedure', 'performed_with', 'procedure'): {
        "code": 1,
        "edges": proc_proc_edges,
        "edge_id": torch.tensor([1] * proc_proc_edges.size()[1]).to(device)
      },
      ('diagnoses', 'given_with', 'procedure'): {
        "code": 2,
        "edges": diag_proc_edges,
        "edge_id": torch.tensor([2] * diag_proc_edges.size()[1]).to(device)
      },
      ('procedure', 'given_with', 'diagnoses'): {
        "code": 3,
        "edges": proc_diag_edges,
        "edge_id": torch.tensor([3] * proc_diag_edges.size()[1]).to(device)
      },
  }

  ## Medicine Data

  prescription_edges = torch.from_numpy(prescriptions_atc3_adj).nonzero().t().contiguous().to(device)
  ddi_edges = torch.from_numpy(ddi_atc3_adj).nonzero().t().contiguous().to(device)

  medical_edges = {
      ('medicine', 'prescribed_with', 'medicine'): {
        "code": 0,
        "edges": prescription_edges,
        "edge_id": torch.tensor([0] * prescription_edges.size()[1]).to(device)
      },
      ('medicine', 'has_ddi_with', 'medicine'): {
        "code": 1,
        "edges": ddi_edges,
        "edge_id": torch.tensor([1] * ddi_edges.size()[1]).to(device)
      },
  }

  ## EHR Data
  patient_info = []

  for subj in range(len(ehr_records)):
    adm_info = []
    for adm in range(1, len(ehr_records[subj])):
      adm_edges = {}

      diagnoses = ehr_records[subj][adm][1]
      procedures = ehr_records[subj][adm][2]
      medicines = ehr_records[subj][adm][3]

      if medicines is not None:
        medicines = [atc3_map_rev_dict[x] for x in medicines]
        adm_edges[('patient', 'prescribed_to', 'medicine')] = {
            "code": 3,
            "edges": torch.stack((
                torch.tensor([num_med_nodes] * len(medicines)),
                torch.tensor(medicines)
            )).to(device),
            "edge_id": torch.tensor([3] * len(medicines)).to(device)
        }
      else: adm_edges[('patient', 'prescribed_to', 'medicine')] = None

      if diagnoses is not None:
        diagnoses = [diag_map_rev_dict[x] for x in diagnoses]
        adm_edges[('patient', 'diagnosed_with', 'diagnosis')] = {
            "code": 4,
            "edges": torch.stack((
                torch.tensor([num_clinical_nodes] * len(diagnoses)),
                torch.tensor(diagnoses)
            )).to(device),
            "edge_id": torch.tensor([4] * len(diagnoses)).to(device)
        }
      else: adm_edges[('patient', 'diagnosed_with', 'diagnosis')] = None

      if procedures is not None:
        procedures = [proc_map_rev_adjusted_dict[x] for x in procedures]
        adm_edges[('patient', 'had_procedure', 'procedure')] = {
            "code": 5,
            "edges": torch.stack((
                torch.tensor([num_clinical_nodes] * len(procedures)),
                torch.tensor(procedures)
            )).to(device),
            "edge_id": torch.tensor([5] * len(procedures)).to(device)
        }
      else: adm_edges[('patient', 'had_procedure', 'procedure')] = None

      adm_info.append(adm_edges)

    patient_info.append(adm_info)

  ############### DDI Knowledge Graph ###############
  ddi_KG = Data().to(device)

  adm_ddi_nodes = torch.unsqueeze(torch.arange(num_med_nodes), 1).to(device)
  adm_ddi_edges = medical_edges[('medicine', 'has_ddi_with', 'medicine')]['edges'].to(device)
  adm_ddi_edge_ids = torch.tensor([0]*len(adm_ddi_edges[0])).to(device)

  ddi_KG.x = adm_ddi_nodes
  ddi_KG.edge_index = adm_ddi_edges.to(torch.int64)
  ddi_KG.edge_type = adm_ddi_edge_ids.to(torch.int64)

  ############# Medical Knowledge Graphs #############
  ehr_KGs = []

  for patient in patient_info:
    patient_KGs = []

    for adm in patient:
      adm_KGs = []
      ############################## Clinical KG ##############################
      clinical_KG = Data().to(device)

      adm_clinical_edges = torch.stack((torch.tensor([]), torch.tensor([]))).to(device)
      adm_clinical_edge_ids = torch.tensor([]).to(device)

      if adm[('patient', 'diagnosed_with', 'diagnosis')] is not None:
        adm_clinical_edges = torch.cat((
            adm_clinical_edges,
            adm[('patient', 'diagnosed_with', 'diagnosis')]['edges'],
            torch.flip(adm[('patient', 'diagnosed_with', 'diagnosis')]['edges'], dims=[0])
        ), 1)
        adm_clinical_edge_ids = torch.cat((
            adm_clinical_edge_ids,
            adm[('patient', 'diagnosed_with', 'diagnosis')]['edge_id'].to(device),
            torch.tensor([6] * len(adm[('patient', 'diagnosed_with', 'diagnosis')]['edge_id'])).to(device)
        ), 0)

        diags = adm[('patient', 'diagnosed_with', 'diagnosis')]['edges'][1]
        diag_diag_edges_filter = clinical_edges[('diagnoses', 'identified_with', 'diagnoses')]['edges'][
            :, torch.isin(clinical_edges[('diagnoses', 'identified_with', 'diagnoses')]['edges'][0], diags)
        ]

        diag_proc_edges_filter = clinical_edges[('diagnoses', 'given_with', 'procedure')]['edges'][
            :, torch.isin(clinical_edges[('diagnoses', 'given_with', 'procedure')]['edges'][0], diags)
        ]

        adm_clinical_edges = torch.cat((
            adm_clinical_edges,
            diag_diag_edges_filter,
            diag_proc_edges_filter,
            torch.flip(diag_diag_edges_filter, dims=[0]),
            torch.flip(diag_proc_edges_filter, dims=[0])
        ), 1)
        adm_clinical_edge_ids = torch.cat((
            adm_clinical_edge_ids,
            torch.tensor([0]*len(diag_diag_edges_filter[0])).to(device),
            torch.tensor([2]*len(diag_proc_edges_filter[0])).to(device),
            torch.tensor([8]*len(diag_diag_edges_filter[0])).to(device),
            torch.tensor([9]*len(diag_proc_edges_filter[0])).to(device),
        ), 0)

      if adm[('patient', 'had_procedure', 'procedure')] is not None:
        adm_clinical_edges = torch.cat((
            adm_clinical_edges,
            adm[('patient', 'had_procedure', 'procedure')]['edges'],
            torch.flip(adm[('patient', 'had_procedure', 'procedure')]['edges'], dims=[0])
        ), 1)
        adm_clinical_edge_ids = torch.cat((
            adm_clinical_edge_ids,
            adm[('patient', 'had_procedure', 'procedure')]['edge_id'].to(device),
            torch.tensor([7]*len(adm[('patient', 'had_procedure', 'procedure')]['edge_id'])).to(device)
        ), 0)

        procs = adm[('patient', 'had_procedure', 'procedure')]['edges'][1]

        proc_proc_edges_filter = clinical_edges[('procedure', 'performed_with', 'procedure')]['edges'][
            :, torch.isin(clinical_edges[('procedure', 'performed_with', 'procedure')]['edges'][0], procs)
        ]

        proc_diag_edges_filter = clinical_edges[('procedure', 'given_with', 'diagnoses')]['edges'][
            :, torch.isin(clinical_edges[('procedure', 'given_with', 'diagnoses')]['edges'][0], procs)
        ]

        adm_clinical_edges = torch.cat((
            adm_clinical_edges,
            proc_proc_edges_filter,
            proc_diag_edges_filter,
            torch.flip(proc_proc_edges_filter, dims=[0]),
            torch.flip(proc_diag_edges_filter, dims=[0])
        ), 1)
        adm_clinical_edge_ids = torch.cat((
            adm_clinical_edge_ids,
            torch.tensor([1]*len(proc_proc_edges_filter[0])).to(device),
            torch.tensor([3]*len(proc_diag_edges_filter[0])).to(device),
            torch.tensor([10]*len(proc_proc_edges_filter[0])).to(device),
            torch.tensor([11]*len(proc_diag_edges_filter[0])).to(device)
        ), 0)

      clinical_KG.x = torch.unsqueeze(torch.arange(num_clinical_nodes+1), 1)
      clinical_KG.edge_index = adm_clinical_edges.to(torch.int64)
      clinical_KG.edge_type = adm_clinical_edge_ids.to(torch.int64)

      ############################### Medical KG ###############################
      medical_KG = Data().to(device)

      adm_medical_edges = torch.stack((torch.tensor([]), torch.tensor([]))).to(device)
      adm_medical_edge_ids = torch.tensor([]).to(device)
      adm_medical_edge_weights = torch.tensor([]).to(device)

      adm_ddi_edges = torch.stack((torch.tensor([]), torch.tensor([]))).to(device)
      adm_ddi_edge_ids = torch.tensor([]).to(device)

      if adm[('patient', 'prescribed_to', 'medicine')] is not None:
        adm_medical_edges = torch.cat((
            adm_medical_edges,
            adm[('patient', 'prescribed_to', 'medicine')]['edges'],
            torch.flip(adm[('patient', 'prescribed_to', 'medicine')]['edges'], dims=[0])
        ), 1)
        adm_medical_edge_ids = torch.cat((
            adm_medical_edge_ids,
            torch.tensor([1] * len(adm[('patient', 'prescribed_to', 'medicine')]['edge_id'])).to(device),
            torch.tensor([2] * len(adm[('patient', 'prescribed_to', 'medicine')]['edge_id'])).to(device)
        ), 0)
        adm_medical_edge_weights = torch.cat((
            adm_medical_edge_weights,
            torch.tensor([1] * len(adm[('patient', 'prescribed_to', 'medicine')]['edge_id'])).to(device),
            torch.tensor([1] * len(adm[('patient', 'prescribed_to', 'medicine')]['edge_id'])).to(device)
        ), 0)

      adm_med_nodes = torch.unsqueeze(torch.arange(num_med_nodes+1), 1)

      medical_KG.x = adm_med_nodes
      medical_KG.edge_index = adm_medical_edges.to(torch.int64)
      medical_KG.edge_type = adm_medical_edge_ids.to(torch.int64)
      medical_KG.edge_weights = adm_medical_edge_weights.to(torch.int64)

      adm_KGs.append(clinical_KG)
      adm_KGs.append(medical_KG)

      patient_KGs.append(adm_KGs)

    ehr_KGs.append(patient_KGs)

  return patient_info, ehr_KGs, ddi_KG, ddi_atc3_adj