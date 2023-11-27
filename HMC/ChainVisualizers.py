import numpy as np
from matplotlib import animation, rc
from IPython.display import HTML

import matplotlib.pyplot as plt



def visualize(chain,output_file,n=300):
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Daniel Trager'), bitrate=1800)
    samples = n

    fig = plt.figure(figsize=(6, 6))
    # chain_output = chain.getOutput().T
    x_lower,x_upper = -5,5#int(np.percentile(chain_output[0],.01))-1,int(np.percentile(chain_output[0],.99))+1
    y_lower,y_upper = -5,5#int(np.percentile(chain_output[1],.01))-1,int(np.percentile(chain_output[1],.99))+1
    i_width = (x_lower, x_upper)
    s_width = (y_lower, y_upper)
    samples_width = (0, samples)
    ax1 = fig.add_subplot(221, xlim=i_width, ylim=samples_width)
    ax2 = fig.add_subplot(224, xlim=samples_width, ylim=s_width)
    ax3 = fig.add_subplot(223, xlim=i_width, ylim=s_width,
                          xlabel='x1',
                          ylabel='x2')
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    line1, = ax1.plot([], [], lw=1)
    line2, = ax2.plot([], [], lw=1)
    line3, = ax3.plot([], [], 'o', lw=2, alpha=.1)
    line4, = ax3.plot([], [], lw=1, alpha=.3)
    line5, = ax3.plot([], [], 'k', lw=1)
    line6, = ax3.plot([], [], 'k', lw=1)
    ax1.set_xticklabels([])
    ax2.set_yticklabels([])
    #path = plt.scatter([], [])
    lines = [line1, line2, line3, line4, line5, line6]

    def init():
        for line in lines:
            line.set_data([], [])
            return lines

    def animate(i):
        # if i >= 200:
        #     i += 300
        current_samples = chain.getOutput()[:i+1].T
        line1.set_data(current_samples[0,:], range(i+1))
        line2.set_data(range(i+1), current_samples[1,:])
        line3.set_data(current_samples[0,:], current_samples[1,:])
        line4.set_data(current_samples[0,:], current_samples[1,:])
        curr_x, curr_y = current_samples[1,-1], current_samples[0,-1]
        line5.set_data([curr_y, curr_y], [curr_x, s_width[1]])
        line6.set_data([curr_y, i_width[1]], [curr_x, curr_x])
        return lines

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=samples, interval=5, blit=True)

    anim.save(output_file, writer=writer)

