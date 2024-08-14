import matplotlib.pyplot as plt
import json
import os
import matplotlib.gridspec as gridspec

k_shots = [5,3,1]
n_effs = [5,3,1]
n_queries = [5,3,1]#[15,10,5,3,1]
n_ways = 257

input_dir = f'log_alpha_voxceleb1_movies_{n_ways}_ways_5s'
output_dir = f'graphs_alpha_voxceleb1_movies_{n_ways}_ways_5s'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for shot in k_shots:
    for n_eff in n_effs:
        # Create subplots
        #fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        #axs=axs.flatten()
        # Create a figure and a GridSpec object
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(2, 6)  # 3 rows, 4 columns

        # Create subplots in the specified grid locations
        ax1 = fig.add_subplot(gs[0, :2])  # Top row, span 2 columns
        ax2 = fig.add_subplot(gs[0, 2:4])  # Top row, span remaining 2 columns
        ax3 = fig.add_subplot(gs[0, 4:]) # Middle row, span 2 columns centered
        ax4 = fig.add_subplot(gs[1, :3])  # Bottom row, span 2 columns left
        ax5 = fig.add_subplot(gs[1, 3:])  # Bottom row, span 2 columns right
        axs = [ax1, ax2, ax3, ax4, ax5]

        for i,n_q in enumerate(n_queries):
            experiment = "k_" + str(shot) +"_neff_" + str(n_eff) + "_nq_" + str(n_q) +".json"
            input_file = input_dir + "/" + experiment
            
            with open(input_file,'r') as f:
                data = json.load(f)
            
            alphas = list(map(int, data["paddle"].keys()))
            accuracies = list(data["paddle"].values())
            simpleshot_accuracy = data["simpleshot"]

            max_accuracy_index = accuracies.index(max(accuracies))
            alpha_max_accuracy = alphas[max_accuracy_index]
            max_paddle_acc = round(data['paddle'][str(alpha_max_accuracy)],2)
            theory_alpha = n_q*n_eff
            if str(theory_alpha) in data['paddle'].keys():
                theory_paddle_acc = round(data['paddle'][str(theory_alpha)],2)
            else:
                alphas_int = [int(i) for i in data['paddle'].keys()]
                theory_paddle_acc =round(data['paddle'][str(min(alphas_int, key=lambda x: abs(x - theory_alpha)))])
            
            if n_eff == 1:
                method_simpleshot = f"d(Ws,Wq),Acc={round(simpleshot_accuracy,2)}"
                method_paddle = "Paddle maj_vote"
            else:
                method_simpleshot = f"Simpleshot,Acc={round(simpleshot_accuracy,2)}"
                method_paddle = "Paddle"

            axs[i].plot(alphas, accuracies, color='blue', marker='o', label=method_paddle)
            axs[i].axhline(y=simpleshot_accuracy, color='red', linestyle='-', label=method_simpleshot)
            
            axs[i].axvline(x=alpha_max_accuracy, color='green', linestyle=':', label=r'$\lambda$'+f'={alpha_max_accuracy},Acc={max_paddle_acc}%', ymax=(accuracies[max_accuracy_index] - min(accuracies)) / (max(accuracies) - min(accuracies)+1e-15))
            axs[i].axvline(x=theory_alpha, color='orange', linestyle=':', label=r'Th. $\lambda$'+f'={theory_alpha},Acc={theory_paddle_acc}%', ymax=(accuracies[max_accuracy_index] - min(accuracies)) / (max(accuracies) - min(accuracies)+1e-15))
            
            axs[i].set_title(f'n_ways={n_ways},k_shot={shot},n_eff={n_eff},n_q={n_q},|Q|={theory_alpha}')
            axs[i].set_xlabel('Alpha')
            axs[i].set_ylabel('Accuracy[%]')
            axs[i].legend()
            axs[i].grid(True)
            output_file = os.path.join(output_dir,"k_" + str(shot) +"_neff_" + str(n_eff)+".png")
            

        # Adjust layout
        #fig.delaxes(axs[-1])
        plt.tight_layout()
        plt.show()
        fig.savefig(output_file)
