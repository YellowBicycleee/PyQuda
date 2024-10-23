import matplotlib.pyplot as plt
import numpy as np

line_color = [None, None, None, 'red', 'orange', 'green', 'blue', None, 'purple']

def draw_table1 (x, x_label, y, y_label, color, table_name) :
    batch_num = color.size
    plt.clf()
    plt.ylim(0, 1)    # y 坐标限制在 0 到 1
    # plt.xlim(0, 10)    # x 坐标限制在 0 到 1
    for i in range(batch_num):
        plt.plot(x[i], y[i], label=f'Nc = {color[i][0]}', marker = 'o')
        plt.xlabel(f'color = {color[i]}')
    plt.title(f'{table_name}')
    plt.xlabel(f'{x_label}')
    plt.ylabel(f'{y_label}')
    
    plt.legend()
    plt.savefig(table_name)
    plt.show()

# x = Nc * g^2
def draw_table2 (x, x_label, y, y_label, color, table_name) :
    batch_num = color.size

    plt.clf()
    plt.title(f'{table_name}')
    plt.ylim(0, 1)    # y 坐标限制在 0 到 1
    plt.xlim(0, 10)    # x 坐标限制在 0 到 10

    # draw real data
    for i in range(batch_num):
        plt.plot(x[i], y[i], label=f'Nc = {color[i][0]}', marker = 'o')
        plt.xlabel(f'color = {color[i]}')
    
    # draw ref data 参考曲线
    ref_x_arr = np.arange(0.1, 10, 0.1)
    ref_y1_arr = ref_y1(ref_x_arr)
    ref_y1_label = '1 / (Nc * g^2)'
    ref_y2_arr = ref_y2(ref_x_arr)
    ref_y2_label = '1 - 2/15 * (Nc * g^2)'
    plt.plot(ref_x_arr, ref_y1_arr, label=f'{ref_y1_label}', linestyle = '--')
    plt.plot(ref_x_arr, ref_y2_arr, label=f'{ref_y2_label}', linestyle = '--')



    plt.xlabel(f'{x_label}')
    plt.ylabel(f'{y_label}')
    
    plt.legend()
    plt.savefig(table_name)
    plt.show()


# ref1 =  1 / (Nc * g^2)
def ref_y1 (Nc_g_square) :
    return 1 / Nc_g_square

# ref2 = 1 - (2/15) * (Nc * g^2)
def ref_y2 (Nc_g_square) :
    return (1 - (2/15) * Nc_g_square)


if __name__ == '__main__' :
    y_label = 'plaq'

    x1_label = 'g^2 = (2 Nc / beta)'
    x2_label = 'Nc g^2 = (2 Nc^2 / beta)'

    y = np.array(
        [
            # Nc = 3
            [   
                1.147157802991e-01, 1.585603132564e-01, 2.051171286958e-01, 2.678084595297e-01, 
                3.397876162723e-01, 4.403599139738e-01, 5.938912129345e-01, 6.562824144072e-01, 
                6.987866850967e-01, 7.561905293716e-01, 8.222968040994e-01, 9.318705584170e-01, 
                9.662732341350e-01, 9.832554038622e-01
            ],
            # Nc = 4
            [
                1.028059151143e-01, 1.397231942198e-01, 1.778222243068e-01, 2.305915054753e-01, 
                2.893196063513e-01, 3.626491417362e-01, 5.511819224108e-01, 6.288483585585e-01, 
                6.765897117363e-01, 7.401385662275e-01, 8.115416240123e-01, 9.279489626180e-01, 
                9.643920893898e-01, 9.823170236193e-01
            ], 
            # Nc = 5
            [
                1.008290405892e-01, 1.349994168001e-01, 1.706684836940e-01, 2.179232849774e-01, 
                2.702512837600e-01, 3.326388610550e-01, 5.135062903250e-01, 6.149362448823e-01, 
                6.659087339247e-01, 7.323156554607e-01, 8.064159392825e-01, 9.261473310685e-01, 
                9.635447072394e-01, 9.818985374135e-01
            ],
            # Nc = 6
            [   
                1.001959102701e-01, 1.339372613641e-01, 1.684496426601e-01, 2.133767098140e-01, 
                2.620971081693e-01, 3.182138997683e-01, 4.030139327903e-01, 6.073458715560e-01, 
                6.600514999296e-01, 7.281598237611e-01, 8.035050553494e-01, 9.252220525842e-01, 
                9.630748651857e-01, 9.816711165184e-01
            ],
            # Nc = 8
            [ 
                1.000847459193e-01, 1.335129938941e-01, 1.672340826799e-01, 2.107238844708e-01, 
                2.563394607850e-01, 3.076878553060e-01, 3.736977664830e-01, 5.992048189478e-01, 
                6.538442352275e-01, 7.236652728097e-01, 8.006708131321e-01, 9.242064293460e-01, 
                9.626223917997e-01, 9.814251224791e-01
            ]
        ]
    )

    # beta = {1.0 1.5 2.0 2.5 3.0 } * Nc * Nc / 3
    Nc = np.array([[3], [4], [5], [6], [8]])

    const = np.array(
        [
            [ 
                0.6, 0.8, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 4, 10, 20, 40
            ]
        ]) 

    beta = (Nc * Nc / 3) * const
    
    # x1 = 2 * Nc / beta
    g_square = 2 * Nc / beta
    # x2 = 2 * Nc^2 / beta
    Nc_g_square = 2 * Nc * Nc / beta

    print(f'beta = \n{beta}')
    print(f'x1 = \n{g_square}')
    print(f'x2 = \n{Nc_g_square}')

    draw_table1(g_square, x1_label, y, y_label, Nc, table_name='plaq ~ g^2')
    draw_table2 (Nc_g_square, x2_label, y, y_label, Nc, table_name='plaq ~ Nc g^2')
