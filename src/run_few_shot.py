import os
import sys 

embeddings = ['embeddings_vox2/jukebox_test.pkl',
              #'embeddings_cn2_vox2/jukebox_test.pkl',
              #'new_embeddings_vox2/cnceleb1_test.pkl',
              #'new_embeddings_cn2_vox2/cnceleb1_test.pkl',
              ]

seeds = [42]

for embs in embeddings:
    use_mean = False
    k_shot = sys.argv[1]
    for seed in seeds:
        out_dir = 'logs_eval_test_repo/log_' + embs.split('/')[0] + '_' + embs.split('/')[1].split('.')[0] + '_seed_' + str(seed)
        os.system(f'python3 src/few_shot.py {embs} {out_dir} {use_mean} {k_shot} {seed}')

exit(0)

for embs in embeddings:
    use_mean = False
    k_shot = sys.argv[2]
    for seed in seeds:
        out_dir = 'logs_eval_10k/log_' + embs.split('/')[0] + '_' + embs.split('/')[1].split('.')[0] + '_seed_' + str(seed)
        os.system(f'python3 src/few_shot.py {embs} {out_dir} {use_mean} {k_shot} {seed}')

for embs in embeddings:
    use_mean = False
    k_shot = sys.argv[3]
    for seed in seeds:
        out_dir = 'logs_eval_10k/log_' + embs.split('/')[0] + '_' + embs.split('/')[1].split('.')[0] + '_seed_' + str(seed)
        os.system(f'python3 src/few_shot.py {embs} {out_dir} {use_mean} {k_shot} {seed}')
