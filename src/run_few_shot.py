import os
import sys 

embeddings = ['embeddings_cn2_vox2/cnceleb1_test.pkl',
              'embeddings_vox2/cnceleb1_test.pkl',
              'embeddings_cn2_vox2/voxceleb1_test.pkl',
              'embeddings_vox2/voxceleb1_test.pkl']


for embs in embeddings:
    out_dir = 'logs_eval/logs_simpleshot_no_mean/log_' + embs.split('/')[0] + '_' + embs.split('/')[1].split('.')[0]
    use_mean = False
    os.system(f'python3 src/few_shot.py {embs} {out_dir} {use_mean}')

for embs in embeddings:
    out_dir = 'logs_eval/logs_simpleshot_mean_subtr/log_' + embs.split('/')[0] + '_' + embs.split('/')[1].split('.')[0]
    use_mean = True
    os.system(f'python3 src/few_shot.py {embs} {out_dir} {use_mean}')