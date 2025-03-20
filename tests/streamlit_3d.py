import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import numpy as np

# 读取CSV文件
def load_data():
    # df = pd.read_csv('data/su6_single_beta.csv')
    # df = pd.read_csv('data/su6_single_lambda.csv')
    df = pd.read_csv('data/su6_start_from_lambda_5.csv')
    return df

# 创建3D图表
def create_3d_surface(df, x_max):
    y = df.columns.astype(float).values  # lambda
    x = np.arange(df.shape[0])  # traj
    x = x[x <= x_max]  # 限制 x 轴的上限
    df = df.iloc[:len(x)]  # 限制数据范围
    X, Y = np.meshgrid(x, y)
    Z = df.values.T  # 转置使得数据对齐

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.update_layout(
        title='3D Surface Plot',
        scene=dict(
            xaxis_title='traj',
            yaxis_title='lambda',
            zaxis_title='plaquette',
            zaxis=dict(range=[0, 1]),  # 限制 z 轴范围在 [0, 1]
            yaxis=dict(range=[0, 5])   # 限制 y 轴范围在 [0, 5]
        )
    )
    return fig

# 创建2D图表
def create_2d_plot(df, axis, value, point_size):
    if axis == 'traj':
        y = df.columns.astype(float).values  # lambda
        z = df.iloc[int(value)].values  # 选择特定的 traj
        fig = go.Figure(data=[go.Scatter(x=y, y=z, mode='lines+markers', marker=dict(size=point_size))])
        fig.update_layout(
            title=f'2D Plot for traj={value}',
            xaxis_title='lambda',
            yaxis_title='plaquette',
            yaxis=dict(range=[0, 1])  # 限制 y 轴范围在 [0, 1]
        )
    else:
        x = np.arange(df.shape[0])  # traj
        z = df.loc[:, str(value)].values  # 选择特定的 lambda
        fig = go.Figure(data=[go.Scatter(x=x, y=z, mode='lines+markers', marker=dict(size=point_size))])
        fig.update_layout(
            title=f'2D Plot for lambda={value}',
            xaxis_title='traj',
            yaxis_title='plaquette',
            yaxis=dict(range=[0, 1])  # 限制 y 轴范围在 [0, 1]
        )
    return fig

def main():
    st.title("3D Surface Plot from CSV Data")
    st.write("This plot visualizes the data in a 3D surface plot.")

    df = load_data()
    # st.write("### CSV Data")
    # st.dataframe(df)

    # 添加滑块控制 x 轴上限
    x_max = st.slider('Select x axis upper limit (traj)', min_value=0, max_value=df.shape[0]-1, value=df.shape[0]-1)

    
    fig3d = create_3d_surface(df, x_max)
    
    st.plotly_chart(fig3d)
    # 添加滑块控制描点大小
    point_size = st.slider('Select point size for 2D plot', min_value=1, max_value=20, value=5)
    # 添加选择框控制显示的二维图轴
    axis = st.selectbox('Select axis for 2D plot', ['traj', 'lambda'])

    if axis == 'traj':
        value = st.slider(f'Select {axis}', min_value=0, max_value=df.shape[0]-1)
    else:
        lambda_options = df.columns.astype(float).values.tolist()
        value = st.selectbox(f'Select {axis}', lambda_options)

    


    st.write("### Select 2D Plot")
    fig2d = create_2d_plot(df, axis, value, point_size)
    st.plotly_chart(fig2d)

if __name__ == '__main__':
    main()
