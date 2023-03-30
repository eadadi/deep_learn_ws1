import matplotlib.pyplot as plt
from part2_baseline import epochs
from part2_optimization import momentum,lr,std
import time
FOLDER="./results/part2/"

x_axis = [i+1 for i in range(epochs)]

def title(s):
    print("=".center(50,"="))
    print(s.replace('_',' ').center(50,"="))
    print("=".center(50,"="))
    return s
def breaksection():
    print("\n")

def plot_2_graphs(results_object_list, labels_list, y_axis_labels,title,ylim=0,save=True):
    i = 0
    for j,results in enumerate(results_object_list):
        plt.clf()
        for y_axis in results:
            plt.plot(x_axis, y_axis, label=labels_list[i])
            i+=1
        plt.xlabel("epochs")
        plt.ylabel(y_axis_labels[j])
        if ylim != 0:
            plt.ylim(ylim[0],ylim[1])
            ylim=0
        plt.grid(True)
        plt.legend()
        plt.title(title.replace('_',' '))
        if not save:
            plt.show()
        else:
            plt.savefig(FOLDER+title+"_"+y_axis_labels[j]+".svg")

def get_labels(name):
    labels = ["{0} Train Err", "{0} Test Err", "{0} Train Acc", "{0} Test Acc"]
    result = []
    for l in labels:
        l=l.format(name)
        result.append(l)
    return result
def p_time(t):
    print(f"Time:{round(t)}".center(50,"="))

total_time = 0
"""
"PART A: compart the best parameters in baseline with/out adam
"       optimization
"""
t = title("SGD_Adam")
from part2_baseline import params, train_and_test_for_params
t0=time.time()
res = train_and_test_for_params(params)
t1=time.time()-t0
total_time+=t1
p_time(t1)

from part2_optimization import activate_optimization
t0=time.time()
result_optimization = activate_optimization()
t1=time.time()-t0
total_time+=t1
p_time(t1)

result_baseline = res[(momentum,lr,std)]
resultsA = [result_baseline[0], result_baseline[2], result_optimization[0], result_optimization[2]]
resultsB = [result_baseline[1], result_baseline[3], result_optimization[1], result_optimization[3]]
results = [resultsA, resultsB]
labels = ["SGD Train Err", "SGD Test Err", "Adam Train Err", "Adam Test Err", "SGD Train Acc", "SGD Test Acc", "Adam Train Acc", "Adam Test Acc"]
ylabels = ["loss","accuracy"]

plot_2_graphs(results, labels, ylabels, ylim=(1,3),title=t)
breaksection()

"""
"PART B: Xavier Initialization
"""
from part2_xavier import activate_xavier
t=title("Xavier_Init")
t0=time.time()
result_xavier = activate_xavier()
t1=time.time()-t0
total_time+=t1
p_time(t1)

results = [[result_xavier[0], result_xavier[2]],[result_xavier[1],result_xavier[3]]]
labels=get_labels("Xavier")

plot_2_graphs(results, labels, ylabels,title=t)
breaksection()

"""
"PART C: Weight decay Initialization
"""
from part2_regularization_decay import activate_decay
t=title("Regularization_decay")
t0=time.time()
results = activate_decay()
t1=time.time()-t0
total_time+=t1
p_time(t1)

results = [[results[0], results[2]],[results[1],results[3]]]
labels=get_labels("Weight Decay")

plot_2_graphs(results, labels, ylabels,title=t)
print()

t=title("Regularization_dropout")
from part2_regularization_dropout import activate_dropout
t0=time.time()
results = activate_dropout()
t1=time.time()-t0
total_time+=t1
p_time(t1)

results = [[results[0], results[2]],[results[1],results[3]]]
labels=get_labels("Dropout")

plot_2_graphs(results, labels, ylabels,title=t)
breaksection()

"""
"PART D: PCA Whitening 
"""
t=title("PCA_Whitening")
from part2_whitening import activate_whitening
t0=time.time()
results = activate_whitening()
t1=time.time()-t0
total_time+=t1
p_time(t1)

results = [[results[0], results[2]],[results[1],results[3]]]
labels=get_labels("Whitening")

plot_2_graphs(results, labels, ylabels,title=t)
breaksection()

"""
"PART E: Varying network widths
"""
from part2_width import activate_width
t=title("Varying_widths")
t0=time.time()
w64, w1024, w4096 = activate_width()
t1=time.time()-t0
total_time+=t1
p_time(t1)

results = [
        [w64[0],w64[2],w1024[0],w1024[2],w4096[0],w4096[2]],
        [w64[1],w64[3],w1024[1],w1024[3],w4096[1],w4096[3]],
        ]
labels = ["w64 Train Err","w64 Test Err", "w1024 Train Err", "w1024 Test Err", "w4096 Train Err", "w4096 Test Err", "w64 Train Acc","w64 Test Acc", "w1024 Train Acc", "w1024 Test Acc", "w4096 Train Acc", "w4096 Test Acc"]

plot_2_graphs(results, labels, ylabels,title=t)
breaksection()

"""
"PART F: Varying network depths
"""
t=title("Varying_depths")
from part2_depth import activate3, activate4, activate10
t0=time.time()
d3 = activate3()
d4 = activate4()
d10 = activate10()
t1=time.time()-t0
total_time+=t1
p_time(t1)

results = [
        [d3[0],d3[2],d4[0],d4[2],d10[0],d10[2]],
        [d3[1],d3[3],d4[1],d4[3],d10[1],d10[3]],
        ]
labels = ["d3 Train Err","d3 Test Err", "d4 Train Err", "d4 Test Err", "d10 Train Err", "d10 Test Err", "d3 Train Acc","d3 Test Acc", "d4 Train Acc", "d4 Test Acc", "d10 Train Acc", "d10 Test Acc"]

plot_2_graphs(results, labels, ylabels,title=t)
breaksection()

title(f"Total_time: {total_time}")
