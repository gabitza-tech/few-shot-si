import torch.nn.functional as F
from utils.paddle_utils import GLASSO,get_one_hot, Logger, most_common_value, top_k_most_common
from tqdm import tqdm
import torch
import time
import numpy as np
from copy import deepcopy
from sklearn.metrics import confusion_matrix
import os
import pickle
import matplotlib.pyplot as plt
from utils.utils import majority_or_original

class KM(object):

    def __init__(self, device, args):#, log_file, args):
        self.device = device
        self.iter = args['iter']
        self.alpha = args['alpha']         # Should be the size of the Query set 
        #self.log_file = log_file
        #self.logger = Logger(__name__, self.log_file)
        if 'maj_vote' in args.keys():
            self.maj_vote = args['maj_vote']
        else:
            self.maj_vote = False
        self.init_info_lists()
        self.covariance_used = "GLASSO"#args['covariance_used']
        self.ckpt_path = "."#args['ckpt_path']
        self.s_use_all_train_set = False#args['s_use_all_train_set']
        self.select_support_elements = False#type(args['select_support_nb_elements'])==int
        self.select_support_nb_elements = False#args['select_support_nb_elements']
        self.predire_une_seule_classe = False#args['predire_une_seule_classe']

    def init_info_lists(self):
        self.timestamps = []
        self.criterions = []
        self.test_acc = []
        self.test_acc_top5 = []
        self.preds_q = []

    def get_logits(self, samples, use_s=True):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]

        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """

        if use_s:
            if self.s_is_diag:
                logits = self.get_logits_avec_S(samples)
            else:
                logits = self.get_logits_avec_S_full(samples) 
                # torch.nan_to_num(torch.logdet(self.s))
                ## TODO : les 0 dans le det à gérer vraiment
        else:
            logits = 1 / 2 * ((self.w.unsqueeze(1) - samples.unsqueeze(2))**2).sum(dim=-1)  # N x n x K (ie n_task x n_query x num_class)    
        
        return logits
    
    def get_logits_avec_S(self,samples):
        logits = (self.w.unsqueeze(1) - samples.unsqueeze(2))**2 * self.s.unsqueeze(1)
        logits = 1/2 * logits.sum(3) - 1/2 * torch.log(self.s.unsqueeze(1)+1e-6).sum(3)
        return logits
    
    def get_logits_avec_S_full(self,samples):
        n_task, n_query = samples.size(0), samples.size(1)
        num_class = self.s.size(1)
        logits = torch.zeros(n_task,n_query,num_class).to(self.device)

        for a in range(n_task):
            for n in range(n_query):
                for k in range(num_class):
                    logits[a,n,k]=torch.matmul((samples[a,n]-self.w[a,k]).unsqueeze(0),torch.matmul(self.s[a,k],samples[a,n]-self.w[a,k]))

        #diff = self.w.unsqueeze(1) - samples.unsqueeze(2)  # N x n x K x C
        #logits = ((diff.square_()).mul_(self.s.unsqueeze(1))).sum(dim=-1)

        return 0.5 * logits - 0.5 * torch.logdet(self.s).unsqueeze(1)

    def init_w(self, support, y_s):
        """
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]

        updates :
            self.w : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        n_tasks = support.size(0)
        one_hot = get_one_hot(y_s).float()
        counts = one_hot.sum(1).view(n_tasks, -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support)
        self.w = weights / counts

    def record_convergence(self, new_time, criterions):
        """
        inputs:
            new_time : scalar
            criterions : torch.Tensor of shape [n_task]
        """
        self.criterions.append(criterions)
        self.timestamps.append(new_time)

    def record_info(self, y_q):
        """
        inputs:
            y_q : torch.Tensor of shape [n_task, n_query] :
        """

        preds_q = self.u.argmax(2).to(torch.device('cpu'))
        y_q = y_q.to(torch.device('cpu'))
        conf = self.u.max(2).values

        # if the file average_confusion_matrix.npy does not exist, create it, otherwise load it
        #cf_matrix = np.zeros((1, a, a))
        predictions, groundtruth, confidences = [], [], []
        for i in range(len(preds_q)):
            predictions += list(preds_q[i])
            groundtruth += list(y_q[i].cpu().numpy())
            confidences += list(conf[i].cpu().numpy())
        
        if self.maj_vote:
            
            preds_q_maj = majority_or_original(preds_q)
            accuracy = (preds_q_maj == y_q).float().mean(1, keepdim=True)
            self.preds_q.append(preds_q_maj)

        else:
            accuracy = (preds_q == y_q).float().mean(1, keepdim=True)
        
        self.test_acc.append(accuracy)

        return groundtruth, predictions, confidences

    def get_logs(self):

        self.criterions = torch.stack(self.criterions, dim=0).cpu().numpy()
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        return {'timestamps': self.timestamps, 'criterions':self.criterions,
                'acc': self.test_acc,'preds_q':self.preds_q}

    def run_task(self, task_dic, all_features_trainset, all_labels_trainset, support_features_params,
                 same_query_size=False, gamma=1.0):
        """
        inputs:
            task_dic : dictionnary with n_tasks few-shot tasks
            shot : scalar, number of shots
        """

        # Extract support and query
        y_s = task_dic['y_s']               # [n_task, shot]
        y_q = task_dic['y_q']               # [n_task, n_query]
        support = task_dic['x_s']           # [n_task, shot, feature_dim]
        query = task_dic['x_q']             # [n_task, n_query, feature_dim]

        if isinstance(self.alpha,str):
            if self.alpha=='adaptatif':
                self.alpha=y_q.shape[1]
            elif self.alpha[:10]=='adaptatif_' and self.alpha[-1]=='%':
                proportion=int(self.alpha[10:-1])/100
                self.alpha=y_q.shape[1]*proportion

        # Transfer tensors to GPU if needed
        support = support.to(self.device).float()
        query = query.to(self.device).float()
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)

        # Run adaptation
        if self.predire_une_seule_classe:
            truth, prediction, confidence = self.run_method_1_classe(support, query, y_s, y_q, all_features_trainset, 
                                                                     all_labels_trainset, gamma, support_features_params)
            logs = {'acc':torch.cat(self.test_acc, dim=1).cpu().numpy()} 
        else:
            truth, prediction, confidence = self.run_method(support, query, y_s, y_q, all_features_trainset, 
                                                            all_labels_trainset, gamma, support_features_params)
            # Extract adaptation logs
            logs = self.get_logs()

        return logs


class PADDLE_GLASSO(KM):

    def __init__(self, device,args):#, log_file, args):
        super().__init__(device=device,args=args)#, log_file=log_file, args=args)

    def A(self, p):
        """
        inputs:

            p : torch.tensor of shape [n_tasks, q_shot, num_class]
                where p[i,j,k] = probability of point j in task i belonging to class k
                (according to our L2 classifier)
        returns:
            v : torch.Tensor of shape [n_task, q_shot, num_class]
        """
        q_shot = p.size(1)
        v = p.sum(1) / q_shot
        return v

    def A_adj(self, v, q_shot):
        """
        inputs:
            V : torch.tensor of shape [n_tasks, num_class]
            q_shot : int
        returns:
            p : torch.Tensor of shape [n_task, q_shot, num_class]
        """
        p = v.unsqueeze(1).repeat(1, q_shot, 1) / q_shot
        return p


    
    def u_update(self, query):
        """
        inputs:
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
         
        updates:
            self.u : torch.Tensor of shape [n_task, n_query, num_class]
        """
        
        n_query = query.size(1)
        logits = self.get_logits(query).detach()

        self.u = (- logits + self.alpha * self.A_adj(self.v, n_query)).softmax(2)


    def v_update(self):
        """
        inputs:
        """
        self.v = torch.log(self.A(self.u) + 1e-6) + 1

    def w_update(self, support, query, y_s_one_hot, gamma):
        """
        Corresponds to w_k updates
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s_one_hot : torch.Tensor of shape [n_task, s_shot, n_ways]


        updates :
            self.w : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        num = (query.unsqueeze(2) * self.u.unsqueeze(3)).sum(1)
        if self.select_support_elements:
            num += gamma*(support.unsqueeze(2) * (y_s_one_hot*self.xi).unsqueeze(3)).sum(1)
            den  = self.u.sum(1) + gamma*(y_s_one_hot*self.xi).sum(1) 
        else:
            num += gamma*(support.unsqueeze(2) * y_s_one_hot.unsqueeze(3)).sum(1)
            den  = self.u.sum(1) + gamma*y_s_one_hot.sum(1) 
        self.w = torch.div(num, den.unsqueeze(2)) 

    def s_update(self, support, query, y_s_one_hot):
        den = ((query.unsqueeze(2) - self.w.unsqueeze(1))**2 * self.u.unsqueeze(3)).sum(1) + 1e-6
        if self.select_support_elements:
            den += ((support.unsqueeze(2) - self.w.unsqueeze(1))**2 * (y_s_one_hot*self.xi).unsqueeze(3)).sum(1)
        else:
            den += ((support.unsqueeze(2) - self.w.unsqueeze(1))**2 * y_s_one_hot.unsqueeze(3)).sum(1)
        num = self.u.unsqueeze(3).sum(1) + y_s_one_hot.unsqueeze(3).sum(1)
        self.s = torch.div(num,den)
    
    def s_init(self, support, y_s_one_hot, y_s, all_features_trainset, all_labels_trainset, support_features_params):
        self.s_is_diag=True
        n_task, n_examples, n_ways = y_s_one_hot.size()
        feature_dim = support.size(2)
        if self.covariance_used == "sans_S":
            self.s = torch.ones(n_task,n_ways,feature_dim).to(self.device)
        elif self.covariance_used in ["S_sans_update", "S_updated"]:
            if self.s_use_all_train_set:
                self.s_init_diag_use_all_train_set(all_features_trainset, all_labels_trainset)
            else:
                self.s_init_diag(support,y_s_one_hot)
        elif self.covariance_used == "S_full":
            if self.s_use_all_train_set:
                self.s_init_full_use_all_train_set(all_features_trainset, all_labels_trainset, support_features_params)
            else:
                self.s_init_full(support,y_s_one_hot)
        elif self.covariance_used == "GLASSO":
            if self.s_use_all_train_set:
                self.s_init_GLASSO_use_all_train_set(all_features_trainset, all_labels_trainset, support_features_params)
            else:
                self.s_init_GLASSO(support, y_s_one_hot, y_s, support_features_params)


    def s_init_diag(self, support, y_s_one_hot):
        den = ((support.unsqueeze(2) - self.w.unsqueeze(1))**2 * y_s_one_hot.unsqueeze(3)).sum(1) # + 1e-6
        num = y_s_one_hot.unsqueeze(3).sum(1)
        self.s = torch.div(num,den)
        self.s_is_diag=True
        n_task, n_query, feature_dim = support.size()
        num_class = y_s_one_hot.size(2)



    def s_init_full(self, support, y_s_one_hot):
        n_task, n_ways = y_s_one_hot.size(0), y_s_one_hot.size(2)
        feature_dim = support.size(2)
        self.s = torch.zeros(n_task,n_ways,feature_dim,feature_dim).to(self.device)
        for a in range(n_task):
            for k in range(n_ways):
                covariance = torch.cov(support[a].T, aweights=y_s_one_hot[a,:,k]) 
                # we transpose support[a] so that rows are the variables and columns are the observations, as required by torch.cov
                # self.s[a,k] = torch.inverse(covariance)  # pas inversible help
                self.s[a,k] = torch.pinverse(covariance)
        self.s_is_diag=False
    
    def s_init_full_use_all_train_set(self, all_features_trainset, all_labels_trainset, features_param):
        save_dir = f'{self.ckpt_path}/features_{features_param}/covariance_matrix/'
        filepath = save_dir+'s_covariance_matrix'+'.plk'
        print(f'loading {filepath}')
        if os.path.isfile(filepath):
            f = open(filepath, 'rb')
            self.s = pickle.load(f)
            f.close()
        else:
            os.makedirs(save_dir, exist_ok=True)

            train_set = all_features_trainset.unsqueeze(0)
            y_all_one_hot = get_one_hot(all_labels_trainset.unsqueeze(0))
            n_task, n_ways = y_all_one_hot.size(0), y_all_one_hot.size(2)
            feature_dim = train_set.size(2)
            self.s = torch.zeros(n_task,n_ways,feature_dim,feature_dim).to(self.device)
            list_classes = ['AM', 'AN', 'NT', 'RE', 'VE']
            for a in range(n_task):
                for k in range(n_ways):
         
                    covariance = torch.cov(train_set[a].T, aweights=y_all_one_hot[a,:,k]) 
                    self.s[a,k] = torch.pinverse(covariance)
                    plt.figure(figsize=(10,10))
                    plt.imshow(abs(covariance), cmap='nipy_spectral')
                    plt.title(f'abs(C_{k})  [classe {list_classes[k]}]')
                    plt.colorbar()
                    plt.savefig(save_dir + f'C_{k}.png')
                    f = open(save_dir + f'C_{k}.plk', 'wb')
                    pickle.dump(covariance, f)
                    f.close()

            f = open(filepath, 'wb')
            pickle.dump(self.s, f)
            f.close()

        self.s_is_diag=False

    def s_init_GLASSO_use_all_train_set(self, all_features_trainset, all_labels_trainset, features_param):
        save_dir = f'{self.ckpt_path}/features_{features_param}/covariance_matrix/'
        lambd_GLASSO = 1e-15
        filepath = save_dir+'s_covariance_matrix_all_train_set_GLASSO_lambda_'+str(lambd_GLASSO)+'.plk'
        print(f'loading {filepath}')
        if os.path.isfile(filepath):
            f = open(filepath, 'rb')
            self.s = pickle.load(f)
            f.close()
        else:
            os.makedirs(save_dir, exist_ok=True)

            train_set = all_features_trainset.unsqueeze(0)
            y_all_one_hot = get_one_hot(all_labels_trainset.unsqueeze(0))
            n_task, n_ways = y_all_one_hot.size(0), y_all_one_hot.size(2)
            feature_dim = train_set.size(2)
            self.s = torch.zeros(n_task,n_ways,feature_dim,feature_dim).to(self.device)
            list_classes = ['AM', 'AN', 'NT', 'RE', 'VE']
            for a in range(n_task):
                for k in range(n_ways):
                    C = torch.cov(train_set[a].T, aweights=y_all_one_hot[a,:,k])
                    S_0 = torch.eye(feature_dim)
                    S = GLASSO(C, S_0, lambd_GLASSO, max_iter=20000, eps=1e-6)
                    self.s[a,k] = S
                    plt.figure(figsize=(10,10))
                    plt.imshow(abs(S), cmap='nipy_spectral')
                    plt.title(f'abs(S_{k})  [classe {list_classes[k]}]')
                    plt.colorbar()
                    plt.savefig(save_dir + f'S_{k}_lambd_{lambd_GLASSO}_all_train_set.png')
                    f = open(save_dir + f'S_{k}.plk', 'wb')
                    pickle.dump(S, f)
                    f.close()

            f = open(filepath, 'wb')
            pickle.dump(self.s, f)
            f.close()

        self.s_is_diag=False

    def s_init_GLASSO(self, support, y_s_one_hot, y_s, features_param):
        save_dir = f'{self.ckpt_path}/features_{features_param}/covariance_matrix/'
        lambd_GLASSO = 1e-7
        filepath = save_dir+'s_covariance_matrix_GLASSO_lambda_'+str(lambd_GLASSO)+'.plk'
        print(f'loading {filepath}')
    
        if False:#os.path.isfile(filepath):
            f = open(filepath, 'rb')
            self.s = pickle.load(f)
            f.close()
        else:
            os.makedirs(save_dir, exist_ok=True)

        
            n_task, n_ways = y_s_one_hot.size(0), y_s_one_hot.size(2)
            feature_dim = support.size(2)
            self.s = torch.zeros(n_task,n_ways,feature_dim,feature_dim).to(self.device)
            for a in tqdm(range(n_task)):
                for k in range(n_ways):
                    features = support[a][y_s[a] == k, ...].to(self.device)
                    C = torch.cov(features.T).to(self.device)
                    
                    if True: #k!= 1:
                        #print("features", features.shape)
                        #C = torch.cov(features.T, aweights=y_s_one_hot[a,:,k])
                        S_0 = torch.eye(feature_dim)
                        S =  GLASSO(C.to(self.device), S_0.to(self.device), lambd_GLASSO, max_iter=200, eps=5e-7)

                        self.s[a,k] = S.to(self.device)
                        # S_new = torch.pinverse(C).to('cpu')
                        # print('sum', torch.sum((S != 0.0)), S.shape[0] * S.shape[1])
                        # S = (S_new * (S != 0.0))
                        # self.s[a,k] = S.to(self.device)
                        #plt.figure(figsize=(10,10))
                        #plt.imshow(abs(S), cmap='nipy_spectral')
                        #plt.title(f'abs(S_{k})  [classe {list_classes[k]}]')
                        #plt.colorbar()
                        #plt.savefig(save_dir + f'S_{k}_lambd_{lambd_GLASSO}_support.png')
                        #f = open(save_dir + f'S_{k}.plk', 'wb')
                        #pickle.dump(S, f)
                        #f.close()
                        #print('ok')

                    else:
                        S=torch.pinverse(C)
                        self.s[a,k] = S
                        plt.figure(figsize=(10,10))
                        plt.imshow(abs(S).cpu().numpy(), cmap='nipy_spectral')
                        plt.title(f'abs(S_{k})  [classe {list_classes[k]}]')
                        plt.colorbar()
                        plt.savefig(save_dir + f'S_{k}_pinv_support.png')
                        f = open(save_dir + f'S_{k}.plk', 'wb')
                        pickle.dump(S, f)
                        f.close()

            #f = open(filepath, 'wb')
            #pickle.dump(self.s, f)
            #f.close()

        self.s_is_diag=False

    def s_init_diag_use_all_train_set(self,all_features_trainset, all_labels_trainset):
        train_set = all_features_trainset.unsqueeze(0)
        y_all_one_hot = get_one_hot(all_labels_trainset.unsqueeze(0))
        n_task, n_ways = y_all_one_hot.size(0), y_all_one_hot.size(2)
        feature_dim = train_set.size(2)
        self.s = torch.zeros(n_task,n_ways,feature_dim).to(self.device)
        for a in range(n_task):
            for k in range(n_ways):
                covariance = torch.cov(train_set[a].T, aweights=y_all_one_hot[a,:,k]) 
                cov_diag = torch.diagonal(covariance)
                self.s[a,k] = 1/cov_diag
        # self.s_init_full_use_all_train_set(all_features_trainset, all_labels_trainset)
        # self.s = torch.diagonal(self.s, dim1=2, dim2=3)
        self.s_is_diag = True



    def xi_update(self, support, y_s_one_hot, p):
        logits = self.get_logits(support)
        tmp = y_s_one_hot * logits
        self.projection_capped_simplex(self.xi - tmp, y_s_one_hot, p)
        
    def projection_capped_simplex(self, vec_to_project, y_s_one_hot, p):
        """
        Projection onto the capped simplex constraint
        :param xi: the vector to project -- numpy array of shape (n_task, shot, num_class)
        :param z: the cap
        :param k: the dimension of the simplex
        :return: the projection of xi on the simplex
        """
        n_task, __, num_class = self.xi.shape
        for i in range(n_task):
            for k in range(num_class):
                index = torch.where(y_s_one_hot[i, :, k] == 1, True, False).squeeze()
                proj = self.projection_fast_capped_simplex_1d(vec_to_project[i, index, k].float(), p)
                self.xi[i, index, k] = proj.to('cuda')

    def projection_capped_simplex_1d(self, y0, p): 
        n = len(y0)
        y, idx = torch.sort(y0)
        x = torch.zeros(n, dtype=torch.float).to('cuda')

        a = n - p
        if (a<=0) or (a>=n) or (y[a] - y[a-1] >= 1): 
            b = a
            x[idx[b:]] = 1
            return x
        for a in range(n):
            for b in range(a+1, n+1):
                gamma = (p + b - n - sum(y[a:b])) / (b-a)
                # Dans l'article, le dernier gamma n'est pas dans l'algo mai il est dans la preuve
                if ((a==0) or (y[a-1]+gamma <= 0)) and (y[a]+gamma > 0) and (y[b-1]+gamma < 1) and ((b==n) or (y[b]+gamma >= 1)):
                    x[idx[b:]] = 1
                    x[idx[a:b]] = y0[idx[a:b]] + gamma
                    return x
        print(y)
    
    def projection_fast_capped_simplex_1d(self, y, p):
        eps = 10**(-4)
        step = 1
        gamma = max(y) 
        zero, one = torch.tensor([0.]).to('cuda'), torch.tensor([1.]).to('cuda')
        compteur=0
        gammas_vus=[]
        while True:
            gammas_vus.append(gamma)
            compteur+=1
            if compteur==100:
                print('Fast capped simplex has not converged after 100 steps on y=')
                print(y)
                input()
            v = y - gamma
            v_dans_intervalle = torch.min(one,torch.max(v,zero))
            omega_at_previous_gamma = p - sum(v_dans_intervalle)
            omega_prime_at_previous_gamma = sum((0<=v)*(v<=1)) 
            if omega_prime_at_previous_gamma == 0: # when we are in a flat region, we use our knowledge that the function is increasing
                if omega_at_previous_gamma < 0:
                    gamma = min(y[y>gamma]) 
                else:
                    gamma = max(y[y<gamma]) 
            elif abs(step*omega_at_previous_gamma/omega_prime_at_previous_gamma) < eps:
                return v_dans_intervalle
            else:
                gamma = gamma - (step*omega_at_previous_gamma/omega_prime_at_previous_gamma)
                if gamma in gammas_vus:
                    step/=2



    def run_method(self, support, query, y_s, y_q, all_features_trainset, all_labels_trainset, gamma, support_features_params):
        """
        Corresponds to the PADDLE inference
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]

        updates :from copy import deepcopy
            self.u : torch.Tensor of shape [n_task, n_query, num_class]         (soft labels)
            self.v : torch.Tensor of shape [n_task, num_class]                  (dual variable)
            self.w : torch.Tensor of shape [n_task, num_class, feature_dim]     (centroids)
        """

        #self.logger.info(" ==> Executing PADDLE with LAMBDA = {}".format(self.alpha))
        y_s_one_hot = get_one_hot(y_s).float()

        n_task, n_examples, n_ways = y_s_one_hot.size()
        feature_dim = support.size(2)
        self.init_w(support=support, y_s=y_s)
        self.v = torch.zeros(n_task, n_ways).to(self.device)
        if self.select_support_elements:
            self.xi = torch.full(y_s_one_hot.size(), self.select_support_nb_elements/(n_task*n_examples)).to(self.device)

        self.s_init(support, y_s_one_hot, y_s, all_features_trainset, all_labels_trainset, support_features_params)

        # y_new = y_s[y_s == 0][:10]
        # support_new = support[y_s == 0][:10]
        # for i in range(1,5):
        #     y_new = torch.cat([y_new, y_s[y_s == i][:10]], dim=-1)
        #     support_new = torch.cat([support_new, support[y_s == i][:10]], dim=0)
        # y_s = y_new.unsqueeze(0)
        # support = support_new.unsqueeze(0)
        # print(y_s.shape, support.shape)
        # y_s_one_hot = get_one_hot(y_s).float()

        for i in tqdm(range(self.iter)): 
            weights_old = deepcopy(self.w.detach())
            t0 = time.time()
            self.u_update(query)
            self.v_update()
            if self.select_support_elements:
                self.xi_update(support, y_s_one_hot, self.select_support_nb_elements)
            self.w_update(support, query, y_s_one_hot, gamma)
            if self.covariance_used == "S_updated":
                self.s_update(support, query, y_s_one_hot)
                        
            weight_diff = (weights_old - self.w).norm(dim=-1).mean(-1)
            criterions = weight_diff
            t1 = time.time()
            self.record_convergence(new_time=t1-t0, criterions=criterions)

        truth, prediction, confidence = self.record_info(y_q=y_q)  
        #print("pred",prediction)
        #print("truth",truth)
        return truth, prediction, confidence
        

    def run_method_1_classe(self, support, query, y_s, y_q, all_features_trainset, all_labels_trainset, gamma, support_features_params):
        y_s_one_hot = get_one_hot(y_s).float()
        n_task, n_examples, n_ways = y_s_one_hot.size()
        n_query = query.size(1)
        vraisemblance = [0 for i in range(n_ways)]
        
        self.s_init(support, y_s_one_hot, y_s,all_features_trainset, all_labels_trainset, support_features_params)
        for k in tqdm(range(n_ways)):
            self.u = torch.zeros(n_task, n_query, n_ways).to(self.device)
            self.u[:,:,k] = 1
            self.w_update(support, query, y_s_one_hot, gamma)
            # if self.covariance_used == "S_full":
            #     logits = self.get_logits_avec_S_full(query)
            # elif self.covariance_used == "sans_S":
            logits = self.get_logits(query)
            vraisemblance[k] = - logits[0,:,k].sum()

        class_to_choose = vraisemblance.index(max(vraisemblance))
        self.u = torch.zeros(n_task, n_query, n_ways).to(self.device)
        self.u[:,:,class_to_choose] = 1

        truth, prediction, confidence = self.record_info(y_q=y_q)  
        return truth, prediction, confidence

class MinMaxScaler(object):
    """MinMax Scaler

    Transforms each channel to the range [a, b].

    Parameters
    ----------
    feature_range : tuple
        Desired range of transformed data.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, query, support, both=False):
        if both:
            features=torch.cat((query,support),dim=1)
            dist = (features.max(dim=1, keepdim=True)[0] - features.min(dim=1, keepdim=True)[0])
            dist[dist==0.] = 1.
            scale = 1.0 /  dist
            ratio = features.min(dim=1, keepdim=True)[0]
        else:
            dist = (query.max(dim=1, keepdim=True)[0] - query.min(dim=1, keepdim=True)[0])
            dist[dist==0.] = 1.
            scale = 1.0 /  dist
            ratio = query.min(dim=1, keepdim=True)[0]
        query.mul_(scale).sub_(ratio)
        support.mul_(scale).sub_(ratio)
        return query, support
    
