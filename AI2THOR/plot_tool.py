import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import re

matplotlib.use('Agg')

COLOR_ALPHA = 0.1
CURVE_WIDTH = 1.5
MARKER_SIZE = 8
matplotlib.rc('font', family='Times New Roman') 
# matplotlib.rc('font', serif='Helvetica Neue') 
# matplotlib.rc('text', usetex='false') 
matplotlib.rcParams.update({'font.size': 16})

figure_handle_list = {}







# ~~~~~~~~~~~~~~~~~~~~DEFINITIONS~~~~~~~~~~~~~~~~~~~
color_list = [
    [239,   0,   0],
    [255, 116,   0],
    [  0, 118, 174],
    [152,  82,  71],
    [158,  99, 181],
    [  0, 161,  59],
    [246, 110, 184],
    [127, 124, 119],
    [194, 189,  44],
]

coloralpha = 0.2

marker_list = [
    's',
    'v',
    '^',
    'h',
    'd',
    'o',
    'p',
    'X',
    '.',
]



# ~~~~~~~~~~~~~~~~~~~~FUNCTIONS~~~~~~~~~~~~~~~~~~~~
def list_dir(folder):
    all_file_list = os.listdir(folder)
    file_prefix_list = []
    for file_name in all_file_list:
        file_prefix = file_name[:-6]
        if file_prefix not in file_prefix_list:
            file_prefix_list.append(file_prefix)
            print(file_prefix)

def read_file(filename_full):
    file_content = []
    with open(filename_full) as file:
        for line in file:
            file_content.append([i for i in line.replace(',', ' ').replace('[', '').replace(']', '').split()])
    test_num, first_test_index, last_test_index = 0, None, None
    for index_i in range(len(file_content)):
        if file_content[index_i][0] == 'test':
            test_num += 1
            last_test_index = index_i
            if first_test_index is None:
                first_test_index = index_i
    if test_num >= 2:
        del file_content[first_test_index:last_test_index]
    return file_content

def handle_figure(title):
    if title in figure_handle_list.keys():
        [fig, subplot] = figure_handle_list[title]
    else:
        fig = plt.figure(figsize=(7, 4), dpi=100)
        subplot = fig.add_subplot(1, 1, 1)
        figure_handle_list[title] = [fig, subplot]
    return [fig, subplot]

def plot_curve(title, x, y, curve_name, color, marker, line_width=CURVE_WIDTH, **kwargs):
    [fig, subplot] = handle_figure(title)
    c = [_ / 255 for _ in color]
    curve = subplot.plot(x, y, color=c,
            linewidth=line_width, marker=marker,
            markevery=max(int(x.shape[-1]/10), 1), markersize=MARKER_SIZE,
            label=curve_name, # '_nolegend_',
            **kwargs,
        )
    
def plot_shade(title, x, y_ave, y_std, colorname, coloralpha):
    [fig, subplot] = handle_figure(title)
    c = [_ / 255 for _ in colorname]
    shade = subplot.fill_between(x, y_ave-y_std, y_ave+y_std, alpha=coloralpha, facecolor=c, edgecolor=c, label='_nolegend_')


def plot_hist(title, x, colorname, coloralpha, **kwargs):
    [fig, subplot] = handle_figure(title)
    c = [_ / 255 for _ in colorname]
    hist = subplot.hist(x, color=c, alpha=coloralpha, **kwargs)
    # plt.hist(data_learned_basis, bins=bin_number, range=[min_v, max_v], density=None, weights=None,
    #     cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical',
    #     rwidth=None, log=False, color=np.array(ptool.color_list[1])/255, alpha=0.3, label='learned', stacked=False)


def plot_legend(title, position):
    if title in figure_handle_list.keys():
        [fig, subplot] = figure_handle_list[title]
        # center upper lower left right
        subplot.legend(loc=position, fontsize=16, labelspacing=0.3)
    else:
        print('Error title name')

def plot_axis_label(title, x_label, y_label):
    if title in figure_handle_list.keys():
        [fig, subplot] = figure_handle_list[title]
        subplot.set_xlabel(x_label, fontsize=20)
        subplot.set_ylabel(y_label, fontsize=20)
        plt.setp(subplot.get_xticklabels(), fontsize=20)
        plt.setp(subplot.get_yticklabels(), fontsize=20)
    else:
        print('Error title name')

def plot_title(title, title_for_display):
    if title in figure_handle_list.keys():
        [fig, subplot] = figure_handle_list[title]
        fig.suptitle(title_for_display, fontsize=26)
    else:
        print('Error title name')
        
def plot_decoration(title, sci_x=False, sci_y=False, grid=True):
    if title in figure_handle_list.keys():
        [fig, subplot] = figure_handle_list[title]
        if sci_x == True:
            subplot.ticklabel_format(axis='x', style='sci', scilimits=(0, 0), useMathText=True)
        if sci_y == True:
            subplot.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
        if grid == True:
            subplot.grid(True, which='major', axis='both', linewidth=0.5, color=[212/255, 212/255, 212/255])
    else:
        print('Error title name')

def plot_save(title, file_name, file_type):
    if title in figure_handle_list.keys():
        [fig, subplot] = figure_handle_list[title]
        filename_full = file_name + file_type
        fig.savefig(filename_full, bbox_inches='tight', pad_inches=0)
    else:
        print('Error title name')


def close_fig(title):
    if title in figure_handle_list.keys():
        [fig, subplot] = figure_handle_list[title]
        plt.close(fig)
        del figure_handle_list[title]
    else:
        print('Error title name')

def show_latex_table(value_array, row_name=None, col_name=None, main_name='', std_array=None):
    '''
    main | col_name
    row_name
    '''
    assert value_array.ndim == 2
    row_num = value_array.shape[0]
    col_num = value_array.shape[1]
    text_list = []
    text_list.append('\\begin{tabular}{%s}' % ('c' * (len(col_name)+1)))
    text_list.append('\t\\toprule')
    # FIRST ROW
    temp_str = '\t'
    for i in range(col_num + 1):
        if i == 0:
            temp_str += '%s & ' % main_name
        else:
            if col_name is None:
                temp_str += ' & '
            else:
                if i == col_num:
                    temp_str += '%s \\\\ ' % (col_name[i-1])
                else:
                    temp_str += '%s & ' % (col_name[i-1])
    text_list.append(temp_str)
    text_list.append('\t\\midrule')
    for row_i in range(row_num):
        temp_str = '\t'
        if row_name != None:
            temp_str += '%s' % (row_name[row_i])
        else:
            temp_str += ' '
        for col_i in range(col_num):
            value_str = ' & %.3f ' % (value_array[row_i, col_i])
            temp_str += value_str
            if isinstance(std_array, np.ndarray):
                temp_str += ' $\\pm$ %.3f' % (std_array[row_i, col_i])
        temp_str += ' \\\\'
        text_list.append(temp_str)
    text_list.append('\t\\bottomrule')
    text_list.append('\\end{tabular}')
    return text_list
