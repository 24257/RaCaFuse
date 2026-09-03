
import sys as _seer_sys
from pathlib import Path as _SeerPath
_seer_sys.path.insert(0, str(_SeerPath(__file__).resolve().parents[1] / "src"))
import seer_security
#%%
import torch;
import numpy as np;
import dill;
import time;
from sklearn.metrics import f1_score, average_precision_score;
import os;
from pathlib import Path;
#%%

records_final_override = None

def create_log_file(folder_name='default'):
    file_path='../logs/{}/{}.txt'.format(folder_name, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return open(file_path, 'w')

def log(file, s):
    print(s)
    if file:
        file.write(s+'\n')
        file.flush()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def multi_label_metric(y_gt, y_pred, y_prob):
    # Jaccard系数
    def jaccard(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if len(union)==0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)
    # 平均精确率
    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score
    # 平均召回率
    def average_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score
    # 平均F1
    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score
    # Macro F1
    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)
    # 平均精确率
    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)
    # Macro F1
    # f1 = f1(y_gt, y_pred)
    # 平均精确率
    prauc = precision_auc(y_gt, y_prob)
    # Jaccard系数
    ja = jaccard(y_gt, y_pred)
    # Precision, Recall, F1
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)
    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)

def ddi_rate_score(med_records, ddi_A):
    all_pairs_cnt=0
    ddi_pairs_cnt=0
    for record in med_records:
        for i in range(len(record)):
            for j in range(i+1, len(record)):
                mi=record[i]; mj=record[j]
                all_pairs_cnt += 1
                if ddi_A[mi][mj]==1 or ddi_A[mj][mi]==1:
                    ddi_pairs_cnt += 1
    if all_pairs_cnt==0:
        return 0
    return ddi_pairs_cnt/all_pairs_cnt

def get_data(dataset_path):
    if records_final_override:
        data = records_final_override
    else:
        data = seer_security.load_pickle(open(Path(dataset_path) / 'records_final.pkl', 'rb'))
    voc = seer_security.load_pickle(open(Path(dataset_path) / 'voc_final.pkl', 'rb'))
    ddi_A=seer_security.load_pickle(open(Path(dataset_path) / 'ddi_A_final.pkl', 'rb'))
    ehr_A=seer_security.load_pickle(open(Path(dataset_path) / 'ehr.pkl', 'rb'))
    return data,voc,ddi_A,ehr_A
	
def load_clinical_state_artifacts(dataset_path):
    labels_path = Path(dataset_path) / "clinical_state_labels.pkl"
    masks_path = Path(dataset_path) / "clinical_state_mask.pkl"
    meta_path = Path(dataset_path) / "clinical_state_meta.pkl"
    if not labels_path.exists():
        return None, None, None
    labels = seer_security.load_pickle(open(labels_path, "rb"))
    masks = seer_security.load_pickle(open(masks_path, "rb"))
    meta = seer_security.load_pickle(open(meta_path, "rb"))
    return labels, masks, meta

def load_clinical_feature_artifacts(dataset_path):
    dataset_path = Path(dataset_path)
    required = {
        "features": dataset_path / "clinical_features.pkl",
        "masks": dataset_path / "clinical_feature_mask.pkl",
        "abnormal": dataset_path / "clinical_feature_abnormal.pkl",
        "delta_hours": dataset_path / "clinical_feature_delta_hours.pkl",
        "meta": dataset_path / "clinical_feature_meta.pkl",
    }
    if not required["features"].exists():
        return None, None, None, None, None
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing clinical feature artifact(s): " + ", ".join(missing))
    return (
        seer_security.load_pickle(open(required["features"], "rb")),
        seer_security.load_pickle(open(required["masks"], "rb")),
        seer_security.load_pickle(open(required["abnormal"], "rb")),
        seer_security.load_pickle(open(required["delta_hours"], "rb")),
        seer_security.load_pickle(open(required["meta"], "rb")),
    )

def build_med_transition_artifacts(records):
    transition_labels = []
    transition_masks = []
    for patient in records:
        history_meds = set()
        for visit in patient:
            current_meds = set(visit[2])
            old_count = len(current_meds & history_meds)
            new_count = len(current_meds - history_meds)
            total = old_count + new_count
            if total == 0:
                transition_labels.append(0)
                transition_masks.append(0)
            elif old_count / total >= 0.7:
                transition_labels.append(0)
                transition_masks.append(1)
            elif new_count / total >= 0.7:
                transition_labels.append(2)
                transition_masks.append(1)
            else:
                transition_labels.append(1)
                transition_masks.append(1)
            history_meds.update(current_meds)
    return transition_labels, transition_masks
#%%
def tensors_to_dynamic_batches(tensors, batch_size, sort=True):
    tensors_=tensors
    sorted_indexes=[i for i in range(len(tensors))]
    if sort:
        all_seq_lens=[(t.shape[0],i) for i,t in enumerate(tensors_)]
        sorted_indexes=[i for l,i in sorted(all_seq_lens)]
        tensors_=[tensors[i] for i in sorted_indexes]

    batches_seq_lens=[]
    batches=[]
    for i in range(0, len(tensors), batch_size):
        seq_lens=[t.shape[0] for t in tensors_[i:i+batch_size]]
        batches_seq_lens.append(seq_lens)
        batch=tensors_[i:i+batch_size]
        max_len=max([t.shape[0] for t in batch])
        # padding
        for j in range(len(batch)):
            t=batch[j]
            if t.shape[0]<max_len:
                padding=torch.zeros((max_len-t.shape[0],*t.shape[1:]), dtype=t.dtype, device=t.device)
                batch[j]=torch.cat((t,padding),dim=0)
        batches.append(torch.stack(batch,dim=0))
    return batches, batches_seq_lens,sorted_indexes


def tensors_to_fixed_batches(tensors, batch_size):
    batches = []
    for i in range(0, len(tensors), batch_size):
        batch = tensors[i:i+batch_size]
        batches.append(torch.stack(batch, dim=0))
    return batches


def list_to_fixed_batches(values, batch_size):
    batches = []
    for i in range(0, len(values), batch_size):
        batches.append(values[i:i+batch_size])
    return batches

#%% test
# import random
# tensors=[]
# for i in range(100):
#     tensors.append(torch.ones(64))
# batches, batches_seq_lens=tensors_to_dynamic_batches(tensors, 16)
#%%
#%%
def _visit_to_tensors(visit, voc_size, device):
    d_voc_size, p_voc_size, m_voc_size = voc_size
    d_tensor=torch.zeros(d_voc_size, dtype=torch.float32)
    diag_codes = torch.as_tensor(visit[0], dtype=torch.long)
    if diag_codes.numel() > 0:
        d_tensor[diag_codes]=1
    p_tensor=torch.zeros(p_voc_size, dtype=torch.float32)
    proc_codes = torch.as_tensor(visit[1], dtype=torch.long)
    if proc_codes.numel() > 0:
        p_tensor[proc_codes]=1
    m_tensor=torch.zeros(m_voc_size, dtype=torch.float32)
    med_codes = torch.as_tensor(visit[2], dtype=torch.long)
    if med_codes.numel() > 0:
        m_tensor[med_codes]=1
    return d_tensor.to(device), p_tensor.to(device), m_tensor.to(device)


def _inserted_visit_record(inserted_visit):
    if isinstance(inserted_visit, dict):
        return inserted_visit["visit"]
    return inserted_visit


def _stack_inserted_visit_tensors(inserted_visits, voc_size, device):
    if not inserted_visits:
        return None
    inserted_diags=[]
    inserted_procs=[]
    inserted_meds=[]
    for inserted_visit in inserted_visits:
        visit = _inserted_visit_record(inserted_visit)
        d_tensor, p_tensor, m_tensor = _visit_to_tensors(visit, voc_size, device)
        inserted_diags.append(d_tensor)
        inserted_procs.append(p_tensor)
        inserted_meds.append(m_tensor)
    return (
        torch.stack(inserted_diags, dim=0),
        torch.stack(inserted_procs, dim=0),
        torch.stack(inserted_meds, dim=0),
    )


def create_dataset(data, voc_size, device, max_visits=-1, start_visit=0, inserted_visits=None):
    d_voc_size, p_voc_size, m_voc_size = voc_size
    diags=[]; procs=[]; meds=[]
    for patient in data:
        v_diags=[]; v_procs=[]; v_meds=[]
        for visit in patient:
            d_tensor=torch.zeros(d_voc_size, dtype=torch.float32)
            d_tensor[torch.tensor(visit[0])]=1
            v_diags.append(d_tensor.to(device))
            p_tensor=torch.zeros(p_voc_size, dtype=torch.float32)
            p_tensor[torch.tensor(visit[1], dtype=torch.long)]=1
            v_procs.append(p_tensor.to(device))
            m_tensor=torch.zeros(m_voc_size, dtype=torch.float32)
            m_tensor[torch.tensor(visit[2])]=1
            v_meds.append(m_tensor.to(device))
        diags.append(torch.stack(v_diags,dim=0))
        procs.append(torch.stack(v_procs,dim=0))
        meds.append(torch.stack(v_meds,dim=0))
    # 对于一个长度为N的病人，采样N个样本，长度为1到N
    s_diags=[]; s_procs=[]; s_meds=[]; s_own_diags=[]; s_own_procs=[]; s_own_meds=[]; labels=[]; ids=[]; visit_ids=[]
    expected_samples = sum(max(0, len(patient) - start_visit) for patient in data)
    if inserted_visits is not None and len(inserted_visits) != expected_samples:
        raise ValueError("inserted_visits is not aligned with target sample count")
    sample_idx=0
    for p_i in range(len(data)):
        for v_i in range(start_visit, len(data[p_i])):
            inserted_tensors = None if inserted_visits is None else _stack_inserted_visit_tensors(
                inserted_visits[sample_idx],
                voc_size,
                device,
            )
            # 注意，tensor切片是共享内存的，所以这里需要clone一下
            if inserted_tensors is None:
                s_diags.append(diags[p_i][:v_i+1].clone())
                s_procs.append(procs[p_i][:v_i+1].clone())
                s_meds.append(meds[p_i][:v_i+1].clone())
                s_own_diags.append(diags[p_i][:v_i+1].clone())
                s_own_procs.append(procs[p_i][:v_i+1].clone())
                s_own_meds.append(meds[p_i][:v_i+1].clone())
            else:
                inserted_diags, inserted_procs, inserted_meds = inserted_tensors
                inserted_diags = inserted_diags.flip(0)
                inserted_procs = inserted_procs.flip(0)
                inserted_meds = inserted_meds.flip(0)
                s_diags.append(torch.cat((diags[p_i][:v_i], inserted_diags, diags[p_i][v_i:v_i+1]), dim=0).clone())
                s_procs.append(torch.cat((procs[p_i][:v_i], inserted_procs, procs[p_i][v_i:v_i+1]), dim=0).clone())
                s_meds.append(torch.cat((meds[p_i][:v_i], inserted_meds, meds[p_i][v_i:v_i+1]), dim=0).clone())
                s_own_diags.append(diags[p_i][:v_i+1].clone())
                s_own_procs.append(procs[p_i][:v_i+1].clone())
                s_own_meds.append(meds[p_i][:v_i+1].clone())
            ids.append(p_i)
            visit_ids.append(v_i)
            sample_idx += 1
    # 屏蔽掉feature中的最后一个时间点的med，因为这个时间点是用来预测的，放到label中
    for m, own_m in zip(s_meds, s_own_meds):
        labels.append(m[-1].clone())
        m[-1]=0
        own_m[-1]=0
    
    if max_visits > 0:
        s_diags = [d[-max_visits:] for d in s_diags]
        s_procs = [p[-max_visits:] for p in s_procs]
        s_meds = [m[-max_visits:] for m in s_meds]
        s_own_diags = [d[-max_visits:] for d in s_own_diags]
        s_own_procs = [p[-max_visits:] for p in s_own_procs]
        s_own_meds = [m[-max_visits:] for m in s_own_meds]

    # 序列取个反向，时间顺序变成由近到远，这样方便后续模型对padding过的序列截断.
    s_diags = [d.flip(0) for d in s_diags]
    s_procs = [p.flip(0) for p in s_procs]
    s_meds = [m.flip(0) for m in s_meds]
    s_own_diags = [d.flip(0) for d in s_own_diags]
    s_own_procs = [p.flip(0) for p in s_own_procs]
    s_own_meds = [m.flip(0) for m in s_own_meds]

    return s_diags,s_procs,s_meds,labels,ids,visit_ids,s_own_diags,s_own_procs,s_own_meds

def _to_visit_tensor(value, device, dtype=torch.float32):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value, dtype=dtype, device=device)
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return torch.empty(0, dtype=dtype, device=device)
        if all(isinstance(item, torch.Tensor) for item in value):
            return torch.stack([item.to(device=device, dtype=dtype) for item in value], dim=0)
        try:
            return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)
        except (TypeError, ValueError):
            return torch.stack(
                [torch.as_tensor(item, dtype=dtype, device=device) for item in value],
                dim=0,
            )
    return torch.as_tensor(value, dtype=dtype, device=device)

def _stack_inserted_artifacts(inserted_visits, artifact_key, device, dtype):
    if not inserted_visits:
        return None
    artifacts = []
    for inserted_visit in inserted_visits:
        if not isinstance(inserted_visit, dict) or artifact_key not in inserted_visit:
            raise ValueError(f"inserted visit is missing clinical artifact: {artifact_key}")
        artifacts.append(_to_visit_tensor(inserted_visit[artifact_key], device=device, dtype=dtype))
    return torch.stack(artifacts, dim=0)


def _create_sequence_artifact_dataset(
    data,
    patient_level_artifacts,
    device,
    max_visits=-1,
    dtype=torch.float32,
    start_visit=0,
    inserted_visits=None,
    inserted_artifact_key=None,
):
    if patient_level_artifacts is None:
        return None
    if len(patient_level_artifacts) != len(data):
        raise ValueError("patient-level clinical artifacts are not aligned with patient count")

    sequences = []
    expected_samples = sum(max(0, len(patient) - start_visit) for patient in data)
    if inserted_visits is not None and len(inserted_visits) != expected_samples:
        raise ValueError("inserted clinical artifacts are not aligned with target sample count")
    sample_idx = 0
    for patient, patient_artifact in zip(data, patient_level_artifacts):
        if len(patient_artifact) != len(patient):
            raise ValueError("patient-level clinical artifacts are not aligned with visit count")
        visit_tensor = _to_visit_tensor(patient_artifact, device=device, dtype=dtype)
        for visit_idx in range(start_visit, len(patient)):
            inserted_tensor = None
            if inserted_visits is not None and inserted_artifact_key is not None:
                inserted_tensor = _stack_inserted_artifacts(
                    inserted_visits[sample_idx],
                    inserted_artifact_key,
                    device,
                    dtype,
                )
            if inserted_tensor is None:
                seq = visit_tensor[:visit_idx + 1].clone()
            else:
                inserted_tensor = inserted_tensor.flip(0)
                seq = torch.cat(
                    (
                        visit_tensor[:visit_idx],
                        inserted_tensor,
                        visit_tensor[visit_idx:visit_idx + 1],
                    ),
                    dim=0,
                ).clone()
            if max_visits > 0:
                seq = seq[-max_visits:]
            sequences.append(seq.flip(0))
            sample_idx += 1
    return sequences

def create_clinical_feature_dataset(
    data,
    clinical_features,
    clinical_feature_masks,
    clinical_feature_abnormal,
    clinical_feature_delta_hours,
    device,
    max_visits=-1,
    start_visit=0,
    inserted_visits=None,
):
    if clinical_features is None:
        return None, None, None, None
    if clinical_feature_masks is None or clinical_feature_abnormal is None or clinical_feature_delta_hours is None:
        raise ValueError("clinical feature inputs require values, masks, abnormal flags, and delta_hours")
    return (
        _create_sequence_artifact_dataset(
            data,
            clinical_features,
            device,
            max_visits=max_visits,
            start_visit=start_visit,
            inserted_visits=inserted_visits,
            inserted_artifact_key="clinical_features",
        ),
        _create_sequence_artifact_dataset(
            data,
            clinical_feature_masks,
            device,
            max_visits=max_visits,
            start_visit=start_visit,
            inserted_visits=inserted_visits,
            inserted_artifact_key="clinical_feature_masks",
        ),
        _create_sequence_artifact_dataset(
            data,
            clinical_feature_abnormal,
            device,
            max_visits=max_visits,
            start_visit=start_visit,
            inserted_visits=inserted_visits,
            inserted_artifact_key="clinical_feature_abnormal",
        ),
        _create_sequence_artifact_dataset(
            data,
            clinical_feature_delta_hours,
            device,
            max_visits=max_visits,
            start_visit=start_visit,
            inserted_visits=inserted_visits,
            inserted_artifact_key="clinical_feature_delta_hours",
        ),
    )

# 从原始数据中构建药物共作用矩阵


def create_directed_weighted_ehr_adj(data, med_voc_size):
    co_counts = torch.zeros((med_voc_size, med_voc_size), dtype=torch.float32)
    med_counts = torch.zeros(med_voc_size, dtype=torch.float32)
    for patient in data:
        for visit in patient:
            meds = list(dict.fromkeys(int(med) for med in visit[2]))
            if not meds:
                continue
            med_idx = torch.tensor(meds, dtype=torch.long)
            med_counts[med_idx] += 1.0
            visit_vector = torch.zeros(med_voc_size, dtype=torch.float32)
            visit_vector[med_idx] = 1.0
            co_counts += torch.outer(visit_vector, visit_vector)
    co_counts.fill_diagonal_(0.0)
    denom = med_counts.view(-1, 1).clamp_min(1.0)
    directed = co_counts / denom
    directed[torch.isnan(directed)] = 0.0
    return directed

def get_med_freq(data,med_voc_size):
    med_freq=torch.zeros(med_voc_size)
    for p in data:
        for v in p:
            m=v[2]
            for i in range(len(m)):
                med_freq[m[i]]+=1
    return med_freq/med_freq.sum()

def create_batches(
    diags,
    procs,
    meds,
    labels,
    ids,
    visit_ids,
    batch_size,
    sort=True,
    clinical_state_targets=None,
    clinical_state_masks=None,
    med_transition_targets=None,
    med_transition_masks=None,
    clinical_features=None,
    clinical_feature_masks=None,
    clinical_feature_abnormal=None,
    clinical_feature_delta_hours=None,
    retrieval_priors=None,
    retrieval_meds=None,
    retrieval_scores=None,
    retrieval_queries=None,
    own_diags=None,
    own_procs=None,
    own_meds=None,
):
    diag_batch,batch_seq_len,sorted_indexes=tensors_to_dynamic_batches(diags, batch_size, sort)
    proc_batch,_,_=tensors_to_dynamic_batches(procs, batch_size, sort)
    med_batch,_,_=tensors_to_dynamic_batches(meds, batch_size, sort)
    own_med_batch = None
    if own_meds is not None:
        if own_diags is None or own_procs is None:
            raise ValueError("own_meds requires own_diags and own_procs")
        own_diags_sorted = [own_diags[i] for i in sorted_indexes]
        own_procs_sorted = [own_procs[i] for i in sorted_indexes]
        own_meds_sorted = [own_meds[i] for i in sorted_indexes]
        own_diag_batch,_,_=tensors_to_dynamic_batches(own_diags_sorted, batch_size, False)
        own_proc_batch,_,_=tensors_to_dynamic_batches(own_procs_sorted, batch_size, False)
        own_med_batch,_,_=tensors_to_dynamic_batches(own_meds_sorted, batch_size, False)
    else:
        own_diag_batch = None
        own_proc_batch = None
    labels_sorted=[labels[i] for i in sorted_indexes]
    label_batch,_,_=tensors_to_dynamic_batches(labels_sorted, batch_size, False)

    aux_batch = None
    aux_mask_batch = None
    if clinical_state_targets is not None:
        aux_targets_sorted = [clinical_state_targets[i] for i in sorted_indexes]
        aux_masks_sorted = [clinical_state_masks[i] for i in sorted_indexes]
        aux_targets_sorted = [
            target if isinstance(target, torch.Tensor) else torch.tensor(target, dtype=torch.float32, device=labels_sorted[0].device)
            for target in aux_targets_sorted
        ]
        aux_masks_sorted = [
            mask if isinstance(mask, torch.Tensor) else torch.tensor(mask, dtype=torch.float32, device=labels_sorted[0].device)
            for mask in aux_masks_sorted
        ]
        aux_batch,_,_=tensors_to_dynamic_batches(aux_targets_sorted, batch_size, False)
        aux_mask_batch,_,_=tensors_to_dynamic_batches(aux_masks_sorted, batch_size, False)

    transition_batch = None
    transition_mask_batch = None
    if med_transition_targets is not None:
        transition_targets_sorted = [med_transition_targets[i] for i in sorted_indexes]
        transition_masks_sorted = [med_transition_masks[i] for i in sorted_indexes]
        transition_batch = []
        transition_mask_batch = []
        for i in range(0, len(transition_targets_sorted), batch_size):
            transition_batch.append(
                torch.tensor(transition_targets_sorted[i:i+batch_size], dtype=torch.long, device=label_batch[0].device)
            )
            transition_mask_batch.append(
                torch.tensor(transition_masks_sorted[i:i+batch_size], dtype=torch.float32, device=label_batch[0].device)
            )

    ids_sorted=[ids[i] for i in sorted_indexes]
    visit_ids_sorted=[visit_ids[i] for i in sorted_indexes]
    ids_batch=[]
    visit_ids_batch=[]
    for i in range(0, len(ids_sorted), batch_size):
        ids_batch.append(ids_sorted[i:i+batch_size])
        visit_ids_batch.append(visit_ids_sorted[i:i+batch_size])

    clinical_feature_batch = None
    clinical_feature_mask_batch = None
    clinical_feature_abnormal_batch = None
    clinical_feature_delta_batch = None
    if clinical_features is not None:
        clinical_features_sorted = [clinical_features[i] for i in sorted_indexes]
        clinical_feature_masks_sorted = [clinical_feature_masks[i] for i in sorted_indexes]
        clinical_feature_abnormal_sorted = [clinical_feature_abnormal[i] for i in sorted_indexes]
        clinical_feature_delta_sorted = [clinical_feature_delta_hours[i] for i in sorted_indexes]
        clinical_feature_batch,_,_=tensors_to_dynamic_batches(clinical_features_sorted, batch_size, False)
        clinical_feature_mask_batch,_,_=tensors_to_dynamic_batches(clinical_feature_masks_sorted, batch_size, False)
        clinical_feature_abnormal_batch,_,_=tensors_to_dynamic_batches(clinical_feature_abnormal_sorted, batch_size, False)
        clinical_feature_delta_batch,_,_=tensors_to_dynamic_batches(clinical_feature_delta_sorted, batch_size, False)

    return (
        diag_batch,
        proc_batch,
        med_batch,
        label_batch,
        ids_batch,
        visit_ids_batch,
        batch_seq_len,
        aux_batch,
        aux_mask_batch,
        transition_batch,
        transition_mask_batch,
        clinical_feature_batch,
        clinical_feature_mask_batch,
        clinical_feature_abnormal_batch,
        clinical_feature_delta_batch,
        tensors_to_fixed_batches([retrieval_priors[i] for i in sorted_indexes], batch_size) if retrieval_priors is not None else None,
        list_to_fixed_batches(sorted_indexes, batch_size),
        tensors_to_fixed_batches([retrieval_meds[i] for i in sorted_indexes], batch_size) if retrieval_meds is not None else None,
        tensors_to_fixed_batches([retrieval_scores[i] for i in sorted_indexes], batch_size) if retrieval_scores is not None else None,
        tensors_to_fixed_batches([retrieval_queries[i] for i in sorted_indexes], batch_size) if retrieval_queries is not None else None,
        own_med_batch,
        own_diag_batch,
        own_proc_batch,
    )

def create_batches_dict(
    data,
    batch_size,
    voc_size,
    device,
    sort=True,
    clinical_state_targets=None,
    clinical_state_masks=None,
    med_transition_targets=None,
    med_transition_masks=None,
    clinical_features=None,
    clinical_feature_masks=None,
    clinical_feature_abnormal=None,
    clinical_feature_delta_hours=None,
    retrieval_priors=None,
    retrieval_meds=None,
    retrieval_scores=None,
    retrieval_queries=None,
    inserted_visits=None,
    max_visits=-1,
    start_visit=0,
):
    clinical_feature_tensors = create_clinical_feature_dataset(
        data,
        clinical_features,
        clinical_feature_masks,
        clinical_feature_abnormal,
        clinical_feature_delta_hours,
        device=device,
        max_visits=max_visits,
        start_visit=start_visit,
        inserted_visits=inserted_visits,
    )
    dataset_tensors = create_dataset(
        data,
        voc_size,
        device=device,
        max_visits=max_visits,
        start_visit=start_visit,
        inserted_visits=inserted_visits,
    )
    tensors=create_batches(
        *dataset_tensors[:6],
        batch_size=batch_size,
        sort=sort,
        clinical_state_targets=clinical_state_targets,
        clinical_state_masks=clinical_state_masks,
        med_transition_targets=med_transition_targets,
        med_transition_masks=med_transition_masks,
        clinical_features=clinical_feature_tensors[0],
        clinical_feature_masks=clinical_feature_tensors[1],
        clinical_feature_abnormal=clinical_feature_tensors[2],
        clinical_feature_delta_hours=clinical_feature_tensors[3],
        retrieval_priors=retrieval_priors,
        retrieval_meds=retrieval_meds,
        retrieval_scores=retrieval_scores,
        retrieval_queries=retrieval_queries,
        own_diags=dataset_tensors[6],
        own_procs=dataset_tensors[7],
        own_meds=dataset_tensors[8],
    )
    return {
        'diag_batches':tensors[0],  # list [batch, dynamic_max_seq_len, size_diag_voc]
        'proc_batches':tensors[1],  # list [batch, dynamic_max_seq_len, size_pro_voc]
        'med_batches':tensors[2],   # list [batch, dynamic_max_seq_len, size_med_voc]
        'label_batches':tensors[3], # list [batch, size_med_voc]
        'ids_batches':tensors[4],   # list [batch, 1], 每个样本对应到data的id
        'visit_ids_batches':tensors[5], # list [batch], target visit index in original patient sequence
        'batch_seq_len':tensors[6], # list list[], batch中每个样本的实际长度
        'clinical_state_batches':tensors[7],
        'clinical_state_mask_batches':tensors[8],
        'med_transition_batches':tensors[9],
        'med_transition_mask_batches':tensors[10],
        'clinical_feature_batches':tensors[11],
        'clinical_feature_mask_batches':tensors[12],
        'clinical_feature_abnormal_batches':tensors[13],
        'clinical_feature_delta_batches':tensors[14],
        'retrieval_prior_batches':tensors[15],
        'sample_index_batches':tensors[16],
        'retrieval_med_batches':tensors[17],
        'retrieval_score_batches':tensors[18],
        'retrieval_query_batches':tensors[19],
        'treatment_episode_score_batches':None,
        'treatment_episode_experience_batches':None,
        'own_med_batches':tensors[20],
        'own_diag_batches':tensors[21],
        'own_proc_batches':tensors[22],
        'n_batches': len(tensors[0])
    }

# %%
