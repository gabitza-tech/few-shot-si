import os
import sys 

embeddings = ['embeddings_cn2_vox2/cnceleb1_dev.pkl',
              'embeddings_vox2/cnceleb1_dev.pkl',
              'embeddings_cn2_vox2/voxceleb1_dev.pkl',
              'embeddings_vox2/voxceleb1_dev.pkl']


for embs in embeddings:
    out_dir = 'logs_tuning_no_mean/log_' + embs.split('/')[0] + '_' + embs.split('/')[1].split('.')[0]
    use_mean = False
    os.system(f'python3 src/alpha_search.py {embs} {out_dir} {use_mean}')

for embs in embeddings:
    out_dir = 'logs_tuning_mean_subtr/log_' + embs.split('/')[0] + '_' + embs.split('/')[1].split('.')[0]
    use_mean = True
    os.system(f'python3 src/alpha_search.py {embs} {out_dir} {use_mean}')