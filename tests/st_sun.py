import streamlit as st
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import my_csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 样本数据
def generate_sample_data():
    Nc = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 12])
    ns = 16
    nt = 16
    y = []
    const = []

    for nc in Nc:
        csv_path = f'./data/plaq_traj_{nc}_{ns}x{nt}.csv'
        if my_csv.file_exists(csv_path):
            const_, lambda_, plaq_ = my_csv.read_csv(csv_path)
            y.append(np.array(plaq_))
            const.append(np.array(const_))
        else:
            y.append(np.random.rand(50))  # 用于示例的随机数据
            const.append(np.linspace(0.1, 5, 50))

    return Nc, const, y

# 参考线函数
def ref_y3(lambda_input):
    return 1 / (2 * lambda_input)

def ref_y4(lambda_input):
    return 1 - (1 / 4) * lambda_input

def create_2d_plots(Nc, const, y, selected_Ncs, point_size):
    beta = [const[i] * Nc[i] * Nc[i] / 3 for i in range(Nc.size)]
    Nc_g_square = [2 * Nc[i] * Nc[i] / beta[i] for i in range(Nc.size)]
    lambda_res = [Nc_g_square[i] / 2 for i in range(Nc.size)]

    fig1 = go.Figure()
    fig2 = go.Figure()

    # 使用颜色映射
    norm = mcolors.Normalize(vmin=min(Nc), vmax=max(Nc))
    cmap = plt.get_cmap('coolwarm')
    colors = [mcolors.rgb2hex(cmap(norm(nc))) for nc in Nc]

    for selected_Nc in selected_Ncs:
        i = Nc.tolist().index(int(selected_Nc))  # 根据 Nc 值查找索引
        fig1.add_trace(go.Scatter(
            x=lambda_res[i],
            y=y[i],
            mode='markers+lines',
            marker=dict(size=point_size),
            line=dict(color=colors[i]),
            name=f'Nc={Nc[i]}'
        ))
        fig2.add_trace(go.Scatter(
            x=const[i],
            y=y[i],
            mode='markers+lines',
            marker=dict(size=point_size),
            line=dict(color=colors[i]),
            name=f'Nc={Nc[i]}'
        ))

    # 添加参考线到 fig1
    lambda_range1 = np.linspace(1, 5, 400)
    lambda_range2 = np.linspace(0, 1.7, 400)
    fig1.add_trace(go.Scatter(
        x=lambda_range1,
        y=ref_y3(lambda_range1),
        mode='lines',
        line=dict(dash='dash', color='black'),
        name='1/(2λ)'
    ))
    fig1.add_trace(go.Scatter(
        x=lambda_range2,
        y=ref_y4(lambda_range2),
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='1 - 1/4λ'
    ))

    fig1.update_layout(
        title='Lambda vs Plaquette',
        xaxis_title='λ',
        yaxis_title='Plaquette',
        yaxis=dict(range=[0, 1])  # 限制 y 轴范围在 [0, 1]
    )
    fig2.update_layout(
        title='Constant vs Plaquette',
        xaxis_title='Constant',
        yaxis_title='Plaquette',
        yaxis=dict(range=[0, 1])  # 限制 y 轴范围在 [0, 1]
    )

    return fig1, fig2

def main():
    st.title("2D Plots for Lambda-Plaquette and Constant-Plaquette Relationships")
    st.write("Interactive 2D plots with selectable Nc values and adjustable point size using Streamlit and Plotly")

    Nc, const, y = generate_sample_data()

    # 添加多选框控制显示的 Nc
    Nc_options = [str(nc) for nc in Nc]
    selected_Ncs = st.multiselect('Select Nc', Nc_options, default=Nc_options)

    # 添加滑块控制点的大小
    point_size = st.slider('Point Size', 1, 20, 5)

    fig1, fig2 = create_2d_plots(Nc, const, y, selected_Ncs, point_size)
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)

    # 显示参考线公式
    st.write("参考线公式：")
    st.latex(r'\frac{1}{2 \lambda}')
    st.latex(r'1 - \frac{1}{4} \lambda')

if __name__ == '__main__':
    main()
