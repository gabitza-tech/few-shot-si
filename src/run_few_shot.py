import os
import sys 

embeddings = ['embeddings_cn2_vox2/jukebox_test.pkl',
              'embeddings_vox2/jukebox_test.pkl']

seeds = [42]#,0,1,1234,10] 

for embs in embeddings:
    use_mean = False
    k_shot = sys.argv[1]
    for seed in seeds:
        out_dir = 'logs_eval/logs_updated_simpleshot_no_mean/log_' + embs.split('/')[0] + '_' + embs.split('/')[1].split('.')[0] + '_seed_' + str(seed)
        os.system(f'python3 src/few_shot.py {embs} {out_dir} {use_mean} {k_shot} {seed}')

exit(0)
for embs in embeddings:
    use_mean = False
    k_shot = sys.argv[2]
    for seed in seeds:
        out_dir = 'logs_eval/logs_updated_simpleshot_no_mean/log_' + embs.split('/')[0] + '_' + embs.split('/')[1].split('.')[0] + '_seed_' + str(seed)
        os.system(f'python3 src/few_shot.py {embs} {out_dir} {use_mean} {k_shot} {seed}')

for embs in embeddings:
    use_mean = False
    k_shot = sys.argv[3]
    for seed in seeds:
        out_dir = 'logs_eval/logs_updated_simpleshot_no_mean/log_' + embs.split('/')[0] + '_' + embs.split('/')[1].split('.')[0] + '_seed_' + str(seed)
        os.system(f'python3 src/few_shot.py {embs} {out_dir} {use_mean} {k_shot} {seed}')
