import matplotlib.pyplot as plt
import numpy as np
def draw_table_mrhs (x, x_label, y, y_label, color, table_name) :
    batch_num = color.size
    plt.clf()
    for i in range(batch_num):
        plt.plot(x[i], y[i], label=f'Nc = {color[i][0]}', marker = 'o')
        plt.xlabel(f'color = {color[i]}')
    plt.title(f'{table_name}')
    plt.xlabel(f'{x_label}')
    plt.ylabel(f'{y_label}')
    plt.ylim(0, 1)    # y 坐标限制在 0 到 1
    plt.legend()
    plt.savefig(table_name)
    plt.show()

if __name__ == '__main__' :
    y_label = 'plaq'

    x1_label = 'g^2 = (2 Nc / beta)'
    x2_label = 'Nc g^2 = (2 Nc^2 / beta)'

    y = np.array(
        [
            # Nc = 3
            [   1.585603132564e-01, 2.051171286958e-01, 2.678084595297e-01, 3.397876162723e-01, 
                4.403599139738e-01, 5.938912129345e-01, 6.562824144072e-01, 6.987866850967e-01, 
                7.561905293716e-01, 8.222968040994e-01
            ],
            # Nc = 4
            [   1.397231942198e-01, 1.778222243068e-01, 2.305915054753e-01, 2.893196063513e-01, 
                3.626491417362e-01, 5.511819224108e-01, 6.288483585585e-01, 6.765897117363e-01, 
                7.401385662275e-01, 8.115416240123e-01 
            ], 
            # Nc = 5
            [
                1.349994168001e-01, 1.706684836940e-01, 2.179232849774e-01, 2.702512837600e-01, 
                3.326388610550e-01, 5.135062903250e-01, 6.149362448823e-01, 6.659087339247e-01, 
                7.323156554607e-01, 8.064159392825e-01
            ],
            # Nc = 6
            [ 
                1.339372613641e-01, 1.684496426601e-01, 2.133767098140e-01, 2.620971081693e-01, 
                3.182138997683e-01, 4.030139327903e-01, 6.073458715560e-01, 6.600514999296e-01, 
                7.281598237611e-01, 8.035050553494e-01
            ],
            # Nc = 8
            [ 
                1.335129938941e-01, 1.672340826799e-01, 2.107238844708e-01, 2.563394607850e-01, 
                3.076878553060e-01, 3.736977664830e-01, 5.992048189478e-01, 6.538442352275e-01,
                7.236652728097e-01, 8.006708131321e-01
            ]
        ]
    )
    # beta = {1.0 1.5 2.0 2.5 3.0 } * Nc * Nc / 3

    Nc = np.array([[3], [4], [5], [6], [8]])

    beta = (Nc * Nc / 3) * np.array([[0.8, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 4 ]]) 
    
    # x1 = 2 * Nc / beta
    x1 = 2 * Nc / beta
    # x2 = 2 * Nc^2 / beta
    x2 = 2 * Nc * Nc / beta

    print(f'beta = \n{beta}')
    print(f'x1 = \n{x1}')
    print(f'x2 = \n{x2}')

    draw_table_mrhs(x1, x1_label, y, y_label, Nc, table_name='plaq ~ g^2')
    draw_table_mrhs(x2, x2_label, y, y_label, Nc, table_name='plaq ~ Nc g^2')
