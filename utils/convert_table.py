import re
import numpy as np
from scipy.stats import rankdata


def create_str_from_final_rank(final_rank_sum, final_rank):
    s = "Final Rank Sum"
    for j in range(len(final_rank_sum)):
        s += " & "
        s += f"{final_rank_sum[j]}"
    s += "\\\\\\hline\n"

    s += "Final Rank"
    for j in range(len(final_rank)):
        s += " & "
        s += f"{final_rank[j]}"
    s += "\\\\\\hline\n"

    return s


def create_str_from_rank(rank_arr, rank_sum, total_rank):
    s = ""
    for i in range(4):
        s += f"RU{i+1}"
        for j in range(rank_arr.shape[1]):
            s += " & "
            s += f"{rank_arr[i, j]}"
        s += "\\\\\n"

    s += f"Rank Sum"
    for j in range(len(rank_sum)):
        s += " & "
        s += f"{rank_sum[j]}"
    s += "\\\\\\hline\n"

    s += f"Rank"
    for j in range(len(total_rank)):
        s += " & "
        s += f"{total_rank[j]}"

    s += "\\\\\\hline\n"

    return s


if __name__ == '__main__':
    with open('../table_data.txt', 'r') as fp:
        txt = fp.read()
    # print(txt)

    txt_modified = txt
    # re.sub(pattern='± [0-9,.]*k', repl='', string=txt_modified)
    txt_modified = re.sub(pattern='\(p *= *[0-9.]+\)', repl='', string=txt_modified)
    txt_modified = re.sub(pattern='± [0-9.]*k', repl='', string=txt_modified)
    txt_modified = re.sub(pattern='RU[0-9]', repl='', string=txt_modified)
    txt_modified = re.sub(pattern='[A-Z]+-RA[0-9]', repl='', string=txt_modified)
    txt_modified = re.sub(pattern='[A-Z]+-RA-E', repl='', string=txt_modified)
    txt_modified = re.sub(pattern='\+', repl='', string=txt_modified)
    txt_modified = re.sub(pattern='\(*[A-Z]*[a-z]*\)*', repl='', string=txt_modified)
    txt_modified = re.sub(pattern='\>1\.0', repl='1001', string=txt_modified)
    txt_modified = re.sub(pattern='1\.0', repl='1000', string=txt_modified)
    txt_modified = re.sub(pattern=' *\n *', repl='\n', string=txt_modified)
    txt_modified = re.sub(pattern='\n\s*\n', repl='\n', string=txt_modified)

    print(txt_modified)

    data = np.fromstring(txt_modified, sep='\t').reshape(-1, 4)

    rank_beam39 = rankdata(data[:11, :].T, axis=1, method='min')
    rank_sum_beam39 = np.sum(rank_beam39, axis=0)
    rank_tot_beam39 = rankdata(rank_sum_beam39, method='min')
    str_beam39 = create_str_from_rank(rank_arr=rank_beam39, rank_sum=rank_sum_beam39, total_rank=rank_tot_beam39)

    rank_beam59 = rankdata(data[11:22, :].T, axis=1, method='min')
    rank_sum_beam59 = np.sum(rank_beam59, axis=0)
    rank_tot_beam59 = rankdata(rank_sum_beam59, method='min')
    str_beam59 = create_str_from_rank(rank_arr=rank_beam59, rank_sum=rank_sum_beam59, total_rank=rank_tot_beam59)

    rank_opf118 = rankdata(data[22:33, :].T, axis=1, method='min')
    rank_sum_opf118 = np.sum(rank_opf118, axis=0)
    rank_tot_opf118 = rankdata(rank_sum_opf118, method='min')
    str_opf118 = create_str_from_rank(rank_arr=rank_opf118, rank_sum=rank_sum_opf118, total_rank=rank_tot_opf118)

    rank_opf300 = rankdata(data[33:44, :].T, axis=1, method='min')
    rank_sum_opf300 = np.sum(rank_opf300, axis=0)
    rank_tot_opf300 = rankdata(rank_sum_opf300, method='min')
    str_opf300 = create_str_from_rank(rank_arr=rank_opf300, rank_sum=rank_sum_opf300, total_rank=rank_tot_opf300)

    rank_truss = rankdata(data[44:, :].T, axis=1, method='min')
    rank_sum_truss = np.sum(rank_truss, axis=0)
    rank_tot_truss = rankdata(rank_sum_truss, method='min')
    str_truss = create_str_from_rank(rank_arr=rank_truss, rank_sum=rank_sum_truss, total_rank=rank_tot_truss)

    rank_tot = np.concatenate([rank_tot_beam39.reshape([1, -1]), rank_tot_beam59.reshape([1, -1]),
                               rank_tot_opf118.reshape([1, -1]), rank_tot_opf300.reshape([1, -1]),
                               rank_tot_truss.reshape([1, -1])],
                              axis=0)
    rank_sum_final = np.sum(rank_tot, axis=0)
    rank_final = rankdata(rank_sum_final, method='min')
    str_final = create_str_from_final_rank(final_rank_sum=rank_sum_final, final_rank=rank_final)
    print(str_final)
