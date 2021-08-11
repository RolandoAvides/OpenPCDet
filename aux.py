import numpy as np
import matplotlib.pyplot as plt
from numpy import load
from yplot.yplot.yplot import read_tfevents
import yplot.yplot.yplot as yplot
import tikzplotlib
import matplotlib.colors as mcolors

plt.style.use("ggplot")


def get_loss_graph(path):
    tb = read_tfevents(path)
    
    
    tag = 'Pedestrian_3d/moderate_R40'
    moderate_3d = np.zeros((81,2))
    i = 0
    for e in tb:
        if e.summary.value[0].tag==tag:
            moderate_3d[i][0] = i
            moderate_3d[i][1] = e.summary.value[0].simple_value
            i+=1
    return moderate_3d
    

    """        
    tag = 'train/loss'
    moderate_3d = np.zeros((81,2))
    i = 0
    epoch = 1
    aux = np.zeros(464)
    for e in tb:
        if e.summary.value[0].tag==tag:
            aux[i] =  e.summary.value[0].simple_value
            i+=1    

            if i==464:
                moderate_3d[epoch-1][0] = epoch
                moderate_3d[epoch-1][1] = np.mean(aux)
                epoch+=1
                i=0
                aux = np.zeros(464)
            
    return moderate_3d
    """

def get_medians(path, tag):
    epochs = 50
    tot_epcohs = 150
    tb = read_tfevents(path)
    metric = np.zeros((150,2))
    i = 0
    for e in tb:
        if e.summary.value[0].tag==tag:
            metric[i][0] = i
            metric[i][1] = e.summary.value[0].simple_value
            i+=1
    print('Final Score of ',tag,' :', np.median(metric[100:,:], axis=0))
    #print(metric[70:,:])
    print('**********\n')

path = '/home/rmoreira/OpenPCDet/experiments/final_eval_3dssd_150_painted_city/home/rmoreira/OpenPCDet/tools/cfgs/kitti_models/3dssd/default/eval/eval_all_default/default/tensorboard_val/events.out.tfevents.1622374713.ctm-deep-01'

get_medians(path,'Car_3d/easy_R40')
get_medians(path,'Car_3d/moderate_R40')
get_medians(path,'Car_3d/hard_R40')

get_medians(path,'Car_bev/easy_R40')
get_medians(path,'Car_bev/moderate_R40')
get_medians(path,'Car_bev/hard_R40')

get_medians(path,'Cyclist_3d/easy_R40')
get_medians(path,'Cyclist_3d/moderate_R40')
get_medians(path,'Cyclist_3d/hard_R40')

#get_medians(path,'Cyclist_aos/easy_R40')
#get_medians(path,'Cyclist_aos/moderate_R40')
#get_medians(path,'Cyclist_aos/hard_R40')

get_medians(path,'Cyclist_bev/easy_R40')
get_medians(path,'Cyclist_bev/moderate_R40')
get_medians(path,'Cyclist_bev/hard_R40')



path = ''
"""
plt.grid(True)
plt.xlabel("epoch")
plt.ylabel("Train Loss")
plt.plot(csv_file[1:,1],label='PV-RCNN',color='tab:blue')


path = '/home/rmoreira/OpenPCDet/pv_rcnn_tensorboards/coco/events.out.tfevents.1619777022.srv-ctm01'
csv_file = get_loss_graph(path)
plt.plot(csv_file[1:,1],label='PV-RCNN + PP-1',color='tab:red')

path = '/home/rmoreira/OpenPCDet/pv_rcnn_tensorboards/city/events.out.tfevents.1623666478.srv-ctm01'
csv_file = get_loss_graph(path)
plt.plot(csv_file[1:,1],label='PV-RCNN + PP-2',color='tab:orange')

plt.legend(loc='upper right')

plt.savefig('/home/rmoreira/plots/test.png')
tikzplotlib.save("/home/rmoreira/plots/test.tex")
""" 
#np.savetxt('/home/rmoreira/plots/pvrcnn-eval-normal.csv', csv_file, delimiter=",")

"""
path = '/home/rmoreira/OpenPCDet/pv_rcnn_tensorboards/normal/events.out.tfevents.1622679719.srv-ctm01'
csv_file = get_loss_graph(path)
np.savetxt('/home/rmoreira/plots/pvrcnn-normal.csv', csv_file, delimiter=",")

path = '/home/rmoreira/OpenPCDet/pv_rcnn_tensorboards/coco/events.out.tfevents.1622590145.srv-ctm01'
csv_file = get_loss_graph(path)
np.savetxt('/home/rmoreira/plots/pvrcnn-coco.csv', csv_file, delimiter=",")
"""
#csv_file = get_loss_graph(path)
#np.savetxt('/home/rmoreira/plots/3dssd-normal.csv', csv_file, delimiter=",")

# para gravar o plot como imagem - caso queiras ver com ficou
#plt.plot(moderate_3d)
#plt.savefig('/home/rmoreira/plots/test.png')
        
#print(tb[0].summary.value[0].tag)
#print(tags)


# python3 -m tensorboard.main --logdir_spec=coco:/home/rmoreira/OpenPCDet/pv_rcnn_tensorboards/coco,normal:/home/rmoreira/OpenPCDet/pv_rcnn_tensorboards/normal
# python3 -m tensorboard.main --logdir_spec=city:,normal:
