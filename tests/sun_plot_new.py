import matplotlib.pyplot as plt
import subprocess
import my_csv
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np

line_color = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
ref_color = ['darkblue', 'm']

# x = Nc * g^2
def draw_table2 (x, x_label, y, y_label, Nc, table_title, file_name = 'Nc_g_square') :
    batch_num = Nc.size
    lw = 1.5
    mk_size = [2 for _ in range(len(Nc))]
    fig = plt.figure(0, figsize=(8, 5))#用来控制图片的大小

    left, bottom, width, height = 0.12, 0.12, 0.8, 0.8
    
    sub1 = fig.add_axes([left, bottom, width, height])
    sub1.axis([0, 10, 0, 1])
    sub1.set_title(f'{table_title}')
    sub1.tick_params(size = 5, labelsize=12, direction='in')
    sub1.grid(visible=True, ls=":")

    # draw real data
    for i in range(batch_num):
        sub1.plot(x[i], y[i], label=r'$N_c=$'+ f'{Nc[i]}', marker = '.', linewidth = lw)
    
    # draw ref data 参考曲线
    ref_x1_arr = np.arange(1.1, 5, 0.1)
    ref_y1_arr = ref_y3(ref_x1_arr)
    ref_y1_label = r'$\frac{1}{2 \lambda}$'
    sub1.plot(ref_x1_arr, ref_y1_arr, label=f'{ref_y1_label}', linestyle = '--', color = ref_color[0], linewidth=2.5)

    ref_x2_arr = np.arange(0.1, 2, 0.1)
    ref_y2_arr = ref_y4(ref_x2_arr)
    ref_y2_label = r'$1 - \frac{1}{4} \lambda$'
    sub1.plot(ref_x2_arr, ref_y2_arr, label=f'{ref_y2_label}', linestyle = '--', color = ref_color[1], linewidth=2.5)

    # 子图
    box = [1.38, 1.55, 0.37, 0.6]
    scale = 4
    left, bottom, width, height = 0.6, 0.5, scale * (box[1] - box[0]) / 2.5, 0.4 * scale * (box[3] - box[2]) / 1
    sub2 = fig.add_axes([left, bottom, width, height])
    sub2.axis(box)
    sub2.tick_params(size=2, labelsize=8, direction='in')
    sub2.set_xlabel(f'{x_label}')
    sub2.set_ylabel(f'{y_label}')
    
    for i in range(batch_num):
        sub2.plot(x[i], y[i], label=f'Nc = {Nc[i]}', marker = '.', linewidth = lw)
    sub2.axvline(x=1.5, color='black', linestyle='--',linewidth=1)
    # 子图结束
    # 大图截取框
    tx0 = box[0]
    tx1 = box[1]
    ty0 = box[2]
    ty1 = box[3]
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    sub1.plot(sx, sy, linestyle='--', linewidth=2, color='black')

    # 使用 mark_inset 连接子图和主图
    mark_inset(sub1, sub2, loc1=2, loc2=4, fc="none", ec="0.5", linestyle='--')

    sub1.axis([0.0, 5, 0, 1])
    sub1.set_xlabel(f'{x_label}')
    sub1.set_ylabel(f'{y_label}')
    # sub1.legend(loc='upper right')
    sub1.legend(loc='lower left')


    plt.savefig(file_name)
    plt.show()


# ref1 =  1 / (Nc * g^2)
def ref_y1 (Nc_g_square) :
    return 1 / Nc_g_square

# ref2 = 1 - (2/15) * (Nc * g^2)
def ref_y2 (Nc_g_square) :
    return (1 - (2/15) * Nc_g_square)

# ref3 = 1 /  (2* lambda)
def ref_y3 (lambda_input) :
    return 1 / (2 * lambda_input)

# ref4 = 1 - 1 / 4 * lambda
def ref_y4 (lamda_input) :
    return (1 - (1 / 4) * lamda_input)

if __name__ == '__main__' :
    ns = 16
    nt = 16

    y_label = 'plaq'

    x1_label = r'$g^2$'
    # x2_label = '$N_c g^2 = 2 \\frac{Nc^2}{beta$}$'
    x2_label = r'$N_c g^2$'

    Nc = np.array([2, 3, 4, 5, 6, 7, 8, 9, 12])

    y = []
    const = []
    for nc in Nc:
        csv_path = f'./data/plaq_traj_{nc}_{ns}x{nt}.csv'

        if my_csv.file_exists(csv_path):
            const_, lambda_, plaq_ = my_csv.read_csv(csv_path)
            y.append(np.array(plaq_))
            const.append(np.array(const_))
        else:
            print(f'file {csv_path} not exists')
            y.append(np.array([]))
            const.append(np.array([]))
    
    # print(f'const = {const}')

    # beta = const * Nc * Nc / 3


    beta = [const[i] * Nc[i] * Nc[i] / 3 for i in range(Nc.size)]
    # x1 = 2 * Nc / beta
    g_square = [2 * Nc[i] / beta[i] for i in range(Nc.size)]
    # x2 = 2 * Nc^2 / beta
    Nc_g_square = [2 * Nc[i] * Nc[i] / beta[i] for i in range(Nc.size)]

    # lambda = N_c g^2 / 2
    lambda_res = [Nc_g_square[i] / 2 for i in range(Nc.size)]
    lambda_label = r'$\lambda$'

    table2_title = r'$\frac{1}{N_c}\mathrm{Tr}[U^{1\times 1}_{P, \mu\nu}(N_c, g^2)]$'
    file_name = 'plaq_lambda.svg'

    draw_table2 (lambda_res, lambda_label, y, y_label, Nc, table_title=table2_title, file_name=file_name)