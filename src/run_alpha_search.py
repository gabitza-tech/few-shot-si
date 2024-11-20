import os
import sys 

embeddings = ['embeddings_cn2_vox2/jukebox_dev.pkl',
              'embeddings_vox2/jukebox_dev.pkl',
              'embeddings_cn2_vox2/cnceleb1_dev.pkl',
              'embeddings_vox2/cnceleb1_dev.pkl']


for embs in embeddings:
    out_dir = 'logs_eval/eval_tuned_nway5/log_' + embs.split('/')[0] + '_' + embs.split('/')[1].split('.')[0]
    os.system(f'python3 src/alpha_search.py {embs} {out_dir}')
    exit(0)
