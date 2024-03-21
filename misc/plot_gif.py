import matplotlib.image as img
import numpy as np


def update_images(num, f_plot_img):
    now_img_path = '/home/wenyongyan/文档/currot-icml_副本/logs/minigrid_A/acrl/ppo_ACRL_EBU_RATIO=0_ACRL_LSP_RATIO=1.0/seed-1/{}_task_evaluation.pdf'.format(num)
    now_img = img.imread(fname=now_img_path)
    f_plot_img.set_data(now_img)
    return [f_plot_img]

ani = FuncAnimation(fig, partial(update_images, f_plot_img=plot_img), np.arange(50), blit=True)

ani.save('evaluation.gif', fps=10)