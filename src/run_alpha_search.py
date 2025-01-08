import os
import sys 

embeddings = ['embeddings_cn2_vox2/jukebox_test.pkl',
              'embeddings_vox2/jukebox_test.pkl',
              'embeddings_cn2_vox2/cnceleb1_test.pkl',
              'embeddings_vox2/cnceleb1_test.pkl',
              ]


for embs in embeddings:
    out_dir = 'logs_eval_final/log_' + embs.split('/')[0] + '_' + embs.split('/')[1].split('.')[0]
    print(out_dir)
    os.system(f'python3 src/alpha_search.py {embs} {out_dir}')
