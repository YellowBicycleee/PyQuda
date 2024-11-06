import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import cycle

def show():
    # plt.clf()
# =============大图===========
    xdata = [i for i in range(1,100)]
    ydata = [(1/i) for i in range(1,100)]
    y2data = [(1/(2*i)) for i in range(1,100)]
    datax=[xdata,xdata]
    datay = [ydata,y2data]
    # plt.clf()
    # plt.legend()
    # plt.show()
    lw = 2
    fig = plt.figure(0, figsize=(5, 5))
    # mpl.rcParams['font.family'] = 'Times New Roman'
#     colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'pink', 'green'])
    colors = cycle(['aqua', 'darkorange'])
    left, bottom, width, height = 0.12, 0.12, 0.8, 0.8
    sub1 = fig.add_axes([left, bottom, width, height])
    sub1.axis([0., xdata[-1], 0, ydata[0]])
    sub1.tick_params(size=5, labelsize=10, direction='in')
    sub1.grid(visible=True, ls=':')

    for i, color in zip(range(2), colors):
        sub1.plot(datax[i], datay[i], color=color, lw=lw,
             label="legend big")

# ============子图===============
    box = [20,25, 0.005, 0.06] # 放大区域
    left, bottom, width, height = 0.5, 0.4, 0.35, 0.35
    sub2 = fig.add_axes([left, bottom, width, height])
    sub2.axis(box)
    sub2.tick_params(size=2, labelsize=8, direction='in')
    sub2.set_xlabel("x1", fontdict={'family': 'Times New Roman',
                                     'weight': 'normal',
                                     'style': 'italic',
                                     'size': 8, })
    sub2.set_ylabel("y1", fontdict={'family': 'Times New Roman',
                                     'weight': 'normal',
                                     'style': 'italic',
                                     'size': 8, })
    for i, color in zip(range(2), colors):
        sub2.plot(datax[i], datay[i], color=color, lw=lw,
                  label='legend samll')
    # ====大图截取框=====
    tx0 = box[0]
    tx1 = box[1]
    ty0 = box[2]
    ty1 = box[3]
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    sub1.plot(sx, sy, "r", linestyle='--', linewidth=1)

    sub1.axis([0.0, xdata[-1], 0, ydata[0]])
    sub1.set_xlabel('x0', fontsize=15)
    sub1.set_ylabel('y0', fontsize=15)
    sub1.set_title(f"title", fontsize=15)
    sub1.legend(loc="upper right")
	# plt.savefig('保存路径')
    plt.show()
    
show()