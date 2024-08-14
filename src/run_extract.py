import os
import sys 

models_dir = sys.argv[1]
eval_lists = sys.argv[2]
num_classes = [7990,5994]

for i,model in enumerate(sorted(os.listdir(models_dir))):
    num_cls = num_classes[i]
    for eval_list in sorted(os.listdir(eval_lists)):
        initial_model = os.path.join(models_dir,model)
        eval_list_path = os.path.join(eval_lists,eval_list)
        output_file = f'embeddings_{model.split(".")[0]}/{eval_list.split("_audios")[0]}.pkl'
        
        os.system(f'python3 src/extract_embeddings.py --n_class {num_cls} --initial_model {initial_model} --eval_list {eval_list_path} --out_file {output_file}')