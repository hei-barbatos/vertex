def argsort(arr):
    # 返回当前位置elem，在序列中由小到大的序号
    sort_index = []
    arr = [[_, arr[_]] for _ in range(len(arr))]
    arr = sorted(arr, key = lambda x: x[1])
    for i, _ in arr:
        sort_index.append(i)
    return sort_index

def calc_auc(y_hat, label):
    '''
    """
    input:
        parma:type: data 二维 y_hat, label (y_hat不需要有序，但要跟label保持对应，需要set(label)==2)
    output:
        return: auc
    """
    '''
    rank_ind = argsort(y_hat)
    pos_cnt = 0
    neg_cnt = 0
    rnm_cnt = 0
    num_cnt = 1
    for ind in rank_ind:
        if label[ind] == 1:
            pos_cnt += 1
            rnm_cnt += num_cnt
        else:
            neg_cnt += 1
        num_cnt += 1

    auc = (rnm_cnt - 0.5 * pos_cnt * (pos_cnt + 1)) / pos_cnt / neg_cnt
    return auc
