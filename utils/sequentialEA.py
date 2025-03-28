import numpy as np
from utils import *
import argparse
import logging


encoder = "gcn"
save_path="seq_dwy"

parser = argparse.ArgumentParser()
logger = logging.getLogger(__name__)
data = "dbp-yago"
name_str = "seq_" + data + "_" + encoder
emb_db_yg=np.load(save_path + "/%s_ins.npy" % (name_str))

data ="dbp-wiki"
name_str = "seq_" + data + "_" + encoder
emb_db_wk=np.load(save_path + "/%s_ins.npy" % (name_str))

parser.add_argument("--evaluate_many", type=bool, default=True)
args = parser.parse_args()


test_index=np.load("test_label_dwy.npy" )

# embs_en_zh=emb_en_zh[test_index[:,[0,1]]]
# embs_en_fr=emb_en_fr[test_index[:,[0,2]]]
# embs_en_ja=emb_en_ja[test_index[:,[0,3]]]

#en-zh
# embs_en_zh=emb_en_zh[test_index[:,0]]
dw_left=emb_db_wk[test_index[:,0]]
dw_right=emb_db_wk[test_index[:,1]]

#en-fr
# ef_left=embs_en_fr[:,0,:]
# ef_right=embs_en_fr[:,1,:]
dy_left=emb_db_yg[test_index[:,0]]
dy_right=emb_db_yg[test_index[:,2]]


if args.evaluate_many:
    top_k = [1,10, 20, 50]
    # top_k = [1]
else:
    top_k = [1, 10]

distance_dw = -sim(dw_left, dw_right, metric="euclidean", normalize=True, csls_k=0)
distance_dy = -sim(dy_left, dy_right, metric="euclidean", normalize=True, csls_k=0)


tasks = div_list(np.array(range(len(test_index))), 10)
pool = multiprocessing.Pool(processes=len(tasks))
reses = list()
for task in tasks:
    distance = [distance_dw[task, :], distance_dy[task, :]]

    reses.append(
        pool.apply_async(multi_cal_rank, (task, distance, top_k, args)))
if args.evaluate_many:
    acc = np.array([0.] * (len(top_k)))
else:
    acc = np.array([[0. for col in range(3)] for row in range(len(top_k))])
pool.close()
pool.join()

best_index = []

mean = 0.
for res in reses:
    (_acc, _mean, _best_index) = res.get()
    acc += _acc
    mean += _mean
    best_index += _best_index
mean /= len(test_index)
best = [test_index[:, 0][i] for i in best_index]
for i in range(len(top_k)):
    # acc[i] = round(acc[i] / len(test), 4)
    acc[i] = acc[i] / len(test_index)
# logger.info(
#     " acc of top {} = {}, meanrank = {:.3f}".format(top_k, acc.tolist(),
#                                                                       mean))
print(
    " acc of top {} = {}, meanrank = {:.3f}".format(top_k, acc.tolist(),
                                                                      mean))


