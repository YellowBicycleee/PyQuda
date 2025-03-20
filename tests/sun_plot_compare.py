import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import pandas as pd
import glob
import os

# 常量定义
PLOT_CONSTANTS = {
    'color_map': {
        '6': 'red',
        '9': 'green',
        '12': 'blue'
    },
    'ref_colors': ['darkblue', 'm'],
    'linewidth': 1.5,
    'figure_size': (8, 5),
    'legend_size': 8
}

class TooManyColorsError(Exception):
    """当发现超过3种颜色时抛出的异常"""
    pass

def read_data_files(data_dir='data_16x16'):
    """读取并处理数据文件"""
    data = {}
    colors_found = set()
    
    # 分别处理from_lambda和single_lambda文件
    pattern1 = os.path.join(data_dir, 'su*_from_lambda_5_full_16.csv')
    pattern2 = os.path.join(data_dir, 'su*_single_lambda_full_16.csv')
    
    for file_path in glob.glob(pattern1) + glob.glob(pattern2):
        filename = os.path.basename(file_path)
        nc = filename.split('_')[0][2:]  # 提取su后面的数字
        is_from_lambda = 'from_lambda' in filename
        
        colors_found.add(nc)
        if len(colors_found) > 3:
            raise TooManyColorsError(f"发现超过3种颜色: {colors_found}")
        
        if nc not in PLOT_CONSTANTS['color_map']:
            raise ValueError(f"未知的颜色编号: {nc}")
            
        # 读取CSV文件
        df = pd.read_csv(file_path)
        plaq_values = df.iloc[-1].values
        lambda_values = df.columns.astype(float)
        
        key = (nc, is_from_lambda)
        if key not in data:
            data[key] = {'lambda': [], 'plaq': []}
            
        # 使用列表append
        for lam, plaq in zip(lambda_values, plaq_values):
            data[key]['lambda'].append(lam)
            data[key]['plaq'].append(plaq)
    
    # 处理完所有文件后，将列表转换为numpy数组并排序
    for key in data:
        lambda_array = np.array(data[key]['lambda'])
        plaq_array = np.array(data[key]['plaq'])
        
        sorted_indices = np.argsort(lambda_array)
        data[key]['lambda'] = lambda_array[sorted_indices]
        data[key]['plaq'] = plaq_array[sorted_indices]
    
    return data

def ref_y3(lambda_input):
    """参考曲线1: 1/(2*lambda)"""
    return 1 / (2 * lambda_input)

def ref_y4(lambda_input):
    """参考曲线2: 1 - lambda/4"""
    return 1 - (1/4) * lambda_input

def create_reference_lines():
    """创建参考线数据"""
    ref_data = {
        'line1': {
            'x': np.arange(1.1, 5, 0.1),
            'label': r'$\frac{1}{2 \lambda}$',
            'color': PLOT_CONSTANTS['ref_colors'][0]
        },
        'line2': {
            'x': np.arange(0.1, 2, 0.1),
            'label': r'$1 - \frac{1}{4} \lambda$',
            'color': PLOT_CONSTANTS['ref_colors'][1]
        }
    }
    ref_data['line1']['y'] = ref_y3(ref_data['line1']['x'])
    ref_data['line2']['y'] = ref_y4(ref_data['line2']['x'])
    return ref_data

def draw_table2(data, table_title, file_name='plaq_lambda.svg'):
    """主绘图函数"""
    fig = plt.figure(0, figsize=PLOT_CONSTANTS['figure_size'])
    
    # 设置主图和子图
    main_ax = plt.axes([0.12, 0.12, 0.8, 0.8])
    main_ax.set_title(table_title)
    main_ax.tick_params(size=5, labelsize=12, direction='in')
    main_ax.grid(visible=True, ls=":")
    
    # 设置子图
    box = [1.38, 1.50, 0.37, 0.6]
    inset_ax = plt.axes([0.6, 0.5, 0.3, 0.3])
    inset_ax.set_xlim(box[0], box[1])
    inset_ax.set_ylim(box[2], box[3])
    inset_ax.tick_params(size=2, labelsize=8, direction='in')
    
    # 按照颜色数值排序
    sorted_data = sorted(data.items(), key=lambda x: int(x[0][0]))
    
    # 绘制数据
    for (nc, is_from_lambda), nc_data in sorted_data:
        color = PLOT_CONSTANTS['color_map'][nc]
        linestyle = '--' if is_from_lambda else '-'
        label = f'$N_c={nc}$' + (' (strong)' if is_from_lambda else ' (weak)')
        
        main_ax.plot(nc_data['lambda'], nc_data['plaq'], 
                    label=label, 
                    color=color,
                    linestyle=linestyle,
                    marker='.', 
                    linewidth=PLOT_CONSTANTS['linewidth'])
        
        inset_ax.plot(nc_data['lambda'], nc_data['plaq'], 
                    color=color,
                    linestyle=linestyle,
                    marker='.', 
                    linewidth=PLOT_CONSTANTS['linewidth'])
    
    # 绘制参考线
    ref_data = create_reference_lines()
    for line in ref_data.values():
        main_ax.plot(line['x'], line['y'], label=line['label'], 
                    linestyle='--', color=line['color'], linewidth=2.5)
    
    # 添加大图截取框
    tx0 = box[0]
    tx1 = box[1]
    ty0 = box[2]
    ty1 = box[3]
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    main_ax.plot(sx, sy, linestyle='--', linewidth=2, color='black')
    
    # 设置子图连接
    mark_inset(main_ax, inset_ax, loc1=2, loc2=4, fc="none", ec="0.5")
    
    # 设置轴标签和范围
    main_ax.set_xlim(0.0, 5)
    main_ax.set_ylim(0, 1)
    main_ax.set_xlabel(r'$\lambda$')
    main_ax.set_ylabel('plaq')
    main_ax.legend(loc='lower left', fontsize=PLOT_CONSTANTS['legend_size'])
    
    plt.savefig(file_name)
    plt.show()

if __name__ == '__main__':
    try:
        # 读取和处理数据
        data = read_data_files()
        
        # 绘图
        draw_table2(
            data,
            r'$\frac{1}{N_c}\mathrm{Tr}[U^{1\times 1}_{P, \mu\nu}(N_c, g^2)]$',
            'compare_strong_weak_coupling_su_6_9_12.pdf'
        )
    except TooManyColorsError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")