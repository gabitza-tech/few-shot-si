import sys
import argparse, glob, os, torch, warnings, time
from ecapa.tools import *
from ecapa.dataLoader import test_loader
from ecapa.ECAPAModel import ECAPAModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "ECAPA_tester")

    parser.add_argument('--initial_model',  type=str,   default="",                                          help='Path of the initial_model')
    parser.add_argument('--n_class', type=int,   default=5994,   help='Number of speakers')
    parser.add_argument('--eval_list',  type=str,   default="/home/gpirlogeanu/git_repos/ECAPA-TDNN/dataset_files/cn_vox2_paths.txt", help='The path of the evaluation list.')
    parser.add_argument('--batch_size', type=int,   default=256,     help='Batch size')
    parser.add_argument('--num_frames', type=int,   default=300,     help='Duration of the input segments, eg: 200 for 2 second')
    parser.add_argument('--n_cpu',      type=int,   default=20,       help='Number of loader threads')
    parser.add_argument('--out_file',      type=str,   default='out_file.pkl',       help='Out pkl file')

    ## Initialization
    warnings.simplefilter("ignore")
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parser.parse_args()
    
    ## Define the data loader
    test_loader = test_loader(test_list=args.eval_list,num_frames=args.num_frames)
    test_dataloader = torch.utils.data.DataLoader(test_loader, batch_size = args.batch_size, shuffle = False, num_workers = args.n_cpu)#, drop_last = True)

    ## Load model
    model = ECAPAModel(**vars(args))
    print("Model %s loaded from previous state!"%args.initial_model)
    model.load_parameters(args.initial_model)

    model.extract_embeddings(test_dataloader,args.out_file)
