import argparse
import copy
import os
import pickle
import sys
# import warnings
from signal import signal, SIGINT
import tempfile
import shutil
from argparse import Namespace

import dash
# import numpy
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
# from dash.exceptions import PreventUpdate
import numpy as np
import plotly.graph_objs as go
import h5py
import networkx as nx

from innovization.vrg_innovization import VRGInnovization
from utils.record_data import INNOVIZATION_DIR, USER_INTERACT_DIR, \
    POWER_LAW_RANK_FILE_PREFIX, CONSTANT_RULE_RANK_FILE_PREFIX, INEQUALITY_RULE_RANK_FILE_PREFIX
# from utils.general import get_repair_agent
from utils.postprocess.statistical_performance import find_pf, calc_hv, calc_igd, calc_igd_plus

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# from file_io import open_file_selection_dialog
from utils.file_io import open_file_selection_dialog


def get_current_gen_data(selected_gen):
    with h5py.File(hdf_file, 'r') as hf:
        all_gen_val = gen_arr
        nearest_gen_value = int(all_gen_val[np.abs(all_gen_val - selected_gen).argmin()])
        gen_key = f'gen{nearest_gen_value}'
        current_gen_data = hf[gen_key]
        obj_label = hf.attrs['obj_label']

        obj = np.array(current_gen_data['F'])
        x = np.array(current_gen_data['X'])
        rank = np.array(current_gen_data['rank'])
        constr = np.array(current_gen_data['G'])

    return nearest_gen_value, x, obj, constr, rank, obj_label


sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
sup = str.maketrans("0123456789.-", "⁰¹²³⁴⁵⁶⁷⁸⁹⋅⁻")
sub_ij = str.maketrans("ij", "ᵢⱼ")
# PLAY_ICON = '\u25B6'
play_icon = '\u23F5'
stop_icon = "\u23F9"
pause_icon = '\u23F8'
refresh_icon = '\u21BA'
save_icon = '\U0001F4BE'

# Parse input arguments
parser = argparse.ArgumentParser("IK-EMO Visualizer")
parser.add_argument('--result-path', type=str, help="Path to the results to be visualized")
parser.add_argument("--port", type=int, default=8050, help="Port to host the dash server")
parser.add_argument("--special-flag", type=str, default=None, help="Any special flag to be passed to the viz portal")
args = parser.parse_args()

if args.result_path is None:
    args.result_path = open_file_selection_dialog(multi_file=False, title="Select optimization history file",
                                                  initialdir=".")

optim_args_file = os.path.join(args.result_path, 'optim_args')
optim_args = 0
with open(optim_args_file, 'r') as fp_args:
    namespace_command = "optim_args = " + fp_args.readline()
    exec(namespace_command)
print(optim_args)
max_gen = optim_args.ngen
print(f"Max. gens = {max_gen}")

hdf_file_original = os.path.join(args.result_path, 'optim_state.hdf5')
# if os.path.exists(hdf_file_original):
temp_dir = tempfile.gettempdir()
temp_path = os.path.join(temp_dir, 'optim_state_temp.hdf5')

hdf_file = temp_path
# hdf_file = hdf_file_original
gen_arr, latest_innov_gen_key, latest_innov_gen, xl, xu, ignore_vars = [], None, None, [], [], []


def read_hdf_file():
    global gen_arr, latest_innov_gen_key, xl, xu, ignore_vars, latest_innov_gen

    shutil.copy2(hdf_file_original, temp_path)
    with h5py.File(hdf_file, 'r') as hf:
        gen_arr = []
        for key in hf.keys():
            gen_no = int(key[3:])
            gen_arr.append(gen_no)
        gen_arr.sort()
        gen_arr = np.array(gen_arr)

        latest_innov_gen = hf.attrs['innov_info_latest_gen']
        latest_innov_gen_key = f"gen{hf.attrs['innov_info_latest_gen']}"
        if hf.attrs['current_gen'] != gen_arr[-1]:
            print("Mismatch in gen numbering")
        xl = hf.attrs['xl']
        xu = hf.attrs['xu']
        ignore_vars = hf.attrs['ignore_vars']
        # KLUGE
        # var_groups = np.array(hf[latest_innov_gen_key]['var_groups'])
        # var_groups = [np.arange(len(xl))]


read_hdf_file()
default_pause_play_icon = pause_icon
if os.path.exists(os.path.join(args.result_path, '.pauserun')):
    default_pause_play_icon = play_icon


if (latest_innov_gen is not None) and (latest_innov_gen > 0):
    with open(os.path.join(args.result_path, INNOVIZATION_DIR,
                           f'innov_{latest_innov_gen_key}.pkl'), 'rb') as innov_fp:
        latest_innov = pickle.load(innov_fp)


def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Closing HDF file. Exiting.')
    # hf.close()
    # print("Closed HDF5 file.")
    if os.path.exists(temp_path):
        os.remove(temp_path)
        print("Removed temporary HDF file.")
    sys.exit(0)


def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

    return fig


def get_hv_fig(hv_list):
    fig = go.Figure(go.Scatter(x=hv_list[:, 0], y=hv_list[:, 1],
                               mode='markers+lines',
                               marker={'line': {'color': 'white'}}))
    fig.add_vline(x=max_gen, line_dash="dot",
                  annotation_text="Termination",
                  line_color='red',
                  annotation_position="right",
                  annotation_textangle=-90)
    fig.update_layout(
        plot_bgcolor='white',
        margin={'r': 50, 't': 10},
        xaxis={
            'title': 'Generations',
            'titlefont': {'size': 18},
            'tickfont': {'size': 16},
            # 'showline': True,
            'linecolor': 'black',
            'zeroline': False,
            'mirror': True,
            'type': 'linear',
            # 'autorange': True,
            'automargin': True,
            'rangemode': 'tozero',
            'showgrid': True,
            'gridcolor': '#EEEEEE'
        },
        yaxis={
            'title': 'Hypervolume',
            'titlefont': {'size': 18},
            'tickfont': {'size': 16},
            # 'showline': True,
            'linecolor': 'black',
            'zeroline': False,
            'mirror': True,
            'type': 'linear',
            'range': [0, 1.5],
            # 'autorange': True,
            'automargin': True,
            'rangemode': 'tozero',
            'showgrid': True,
            'gridcolor': '#EEEEEE'
        }
    )

    return fig


def update_hv_progress(gen_list):
    """Create a HV figure to denote optimization progress."""
    # pf_list = []

    # Calculate nadir point from final gen pf
    nearest_gen_value, x, obj, constr, rank, obj_label = get_current_gen_data(max(gen_list))
    # final_pf = find_pf(obj[rank == 0])
    final_pf = obj[rank == 0]
    ideal_point = np.min(final_pf, axis=0)
    nadir_point = np.max(final_pf, axis=0)

    min_val = ideal_point
    max_val = nadir_point

    hv_list = []
    for gen in gen_list:
        nearest_gen_value, x, obj, constr, rank, obj_label = get_current_gen_data(gen)
        pf_gen = obj[rank == 0]
        pf_normalized = (pf_gen - min_val) / (max_val - min_val)
        hv_gen = calc_hv(pf_normalized, ref_point=np.array([1.1, 1.1]))
        hv_list.append([gen, hv_gen])

    hv_list = np.array(hv_list)

    return hv_list


def get_gen_slider_steps(gen_list):
    gen_slider_steps = {str(int(gen)): '' for gen in gen_list}

    return gen_slider_steps


signal(SIGINT, handler)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', 'static/reset.css',
                        'https://fonts.googleapis.com/icon?family=Material+Icons']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

config = {"toImageButtonOptions": {"width": None, "height": None, "format": 'svg'}}
config_heatmap = {"toImageButtonOptions": {"width": None, "height": None, "format": 'svg'}, 'editable': True}

optim_progress_div = html.Div(id="loading-output-2")
prob_options = [{'label': 'Truss', 'value': 'Truss'}, {'label': 'OPF', 'value': 'OPF'}]
# list_dir = next(os.walk('problems'))[1]
# for d in list_dir:
#     prob_options.append({'label': d, 'value': d})
# obj_options = [
#     {'label': 'y1', 'value': 'y1'},
#     {'label': 'y2', 'value': 'y2'},
#     {'label': 'y3', 'value': 'y3'},
#     {'label': 'y4', 'value': 'y4'},
#     {'label': 'y5', 'value': 'y5'},
#     {'label': 'y6', 'value': 'y6'},
#     {'label': 'y7', 'value': 'y7'},
#     {'label': 'y8', 'value': 'y8'},
#     {'label': 'y9', 'value': 'y9'},
# ]

# Define HTML layout
html_layout = [
    html.Div([
        html.H1(children='IK-EMO Visualizer v1.0', style={'font-weight': 'normal'}),
    ],
        style={'padding': '30px 0px 20px 0px', 'background-color': '#5F4F93',  # '#059862',
               'margin': '0px 0px 20px 0px',
               'border-bottom': '1px #EBEDEF solid',
               # 'font-family': "Arial",
               'color': 'white', 'text-align': 'center',
               # 'position': 'sticky', "top": '0'
               },
    ),
    # Optimization controls
    html.Div([
        html.Div([html.H2(children='IK-EMO Controls', className='widgetTitle')],
                 style={'color': '#3C4B64', 'width': '20%', 'font-weight': 'normal', 'display': 'inline-block',
                        'vertical-align': 'middle'}),

        html.Div([
            html.Div([
                html.Button(default_pause_play_icon, id='pause-continue-optimization', className="button3",
                            title="Pause/continue run",
                            style={'width': '10%', 'margin': '0px 20px 0px 0px', 'display': 'inline-block',
                                   'font-size': '30px', 'border': 'none'}),
                html.Button(refresh_icon, id='refresh-data', className='button', title="Update data",
                            style={'width': '10%', 'margin': '0px 20px 0px 0px', 'display': 'inline-block',
                                   'font-size': '40px', 'border': 'none'}),
                html.Button(save_icon, id='save-data', className='button', title="Save changes",
                            style={'width': '10%', 'margin': '0px 20px 0px 0px', 'display': 'inline-block',
                                   'font-size': '25px', 'border': 'none'}),
                dcc.ConfirmDialog(id='confirm-write',
                                  message='This will write the results to disk. Continue?'),
            ], style={'display': 'inline-block', 'width': '75%', 'vertical-align': 'middle',
                      'padding': '0px 0 0 0'}),

            # html.Div([
                # dcc.Loading(
                #     id="optim-loading",
                #     children=[html.Div([optim_progress_div])],
                #     type="circle",
                # ),
                # html.Div([
                # ], id='optim-rogress', style={'text-align': 'center'}),
                # dcc.Interval(
                #     id='interval-component',
                #     interval=3 * 1000,  # in milliseconds
                #     n_intervals=0
                # )
            # ], style={'display': 'inline-block', 'width': '15%', 'vertical-align': 'middle'}),

        ], style={'width': '80%', 'display': 'inline-block'}
        )
    ],
        style={'width': '95.5%', 'display': 'inline-block', 'vertical-align': 'middle',
               'padding': '0px 0px 0px 40px',
               'border': '1px solid #969696', 'border-radius': '5px',
               'margin': '0px 0px 20px 20px',
               'background-color': 'white'}
    ),

    html.Div([
        # Historical data
        html.Div([
            html.Div([
                html.Div([
                    html.H2(children='Optimization progress', id='hv-evolution-heading',
                            style={'display': 'inline-block'}, className='widgetTitle'),
                    dcc.Interval(id='optim-progress-update-interval',
                                 interval=20 * 1000,  # in milliseconds
                                 n_intervals=0),
                    html.Div([
                        html.Div([], className='circle--outer'),
                        html.Div([], className='circle--inner'),
                    ], className="video__icon", style={'display': 'inline-block'}),
                ]),
                html.Div([
                    dcc.Graph(id='hv-evolution-graph', figure=get_hv_fig(update_hv_progress(gen_arr)),
                              hoverData={'points': [{'customdata': ''}]}, config=config,
                              style={'width': '100%'})
                ], style={'width': '100%', 'padding-top': '20px'})
            ], style={'width': '100%', 'display': 'inline-block', 'padding': '0 0 0 20px'})
        ], style={'width': '29%', 'display': 'inline-block', 'float': 'center',
                  'padding': '20px 20px 20px 20px',
                  'vertical-align': 'top', 'background-color': 'white',
                  'border': '1px solid #969696', 'border-radius': '5px', 'margin': '0px 0px 20px 20px',
                  'overflow': 'scroll', 'height': '83%'}),
        html.Div([
            html.Div([html.H2(children='Scatter plot', id='scatter-heading', className='widgetTitle',
                              style={'display': 'inline-block', 'width': '50%'}),
                      html.Button(children='Get alternate solution', id="similarSolutionButton", title='Mark similar solutions',
                                  style={'margin': '0 0px 0 0', 'display': 'inline-block'})
                      ],
                     style={'padding': '0px 0px 0px 20px', 'color': '#3C4B64'}),
            html.Div([dcc.Graph(id='objective-space-scatter',
                                hoverData={'points': [{'customdata': ''}]}, config=config)],
                     style={'padding': '20px 20px 0px 20px',  # 'background-color': 'white',
                            'margin': '0px 0px 20px 0px', 'border-bottom': '1px #EBEDEF solid',
                            'background-color': 'white'},
                     ),

            html.Div([
                html.H6(children='Generation', id='generation-no', style={'display': 'inline-block'}),
                html.Div([
                    html.Button(children=play_icon, id="playScatter", title='Play',
                                style={'margin': '0 0px 0 0', 'font-size': '20px', 'border': 'none',
                                       'padding-right': '0px'}),
                    html.Button(children=stop_icon, id="stopScatter", title='Pause',
                                style={'margin': '0 0px 0 0', 'font-size': '20px', 'border': 'none',
                                       'padding-left': '10px'})
                ], style={'margin': '0 0 0 20px', 'display': 'inline-block'}),
            ], style={'padding': '0px 20px 0px 20px'}),

            html.Div([
                dcc.Slider(
                    id='cross-filter-gen-slider',
                    min=min(gen_arr),
                    max=max(gen_arr),
                    value=max(gen_arr),
                    step=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                    marks=get_gen_slider_steps(gen_arr),
                    # marks={str(int(gen)): str(int(gen)) for gen in gen_arr}
                )
            ], style={'padding': '0px 20px 0px 0px'}, id='slider-div'),
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px 20px 0px 20px',
                  'vertical-align': 'top', 'border': '1px solid #969696', 'border-radius': '5px',
                  'margin': '0px 20px 20px 20px', 'background-color': 'white'}),
        # Padding: top right bottom left
        html.Div([
            html.Div([html.H2(children='Parallel coordinate plot (PCP)', id='pcp-heading', className='widgetTitle')],
                     style={'padding': '0px 20px 20px 20px', 'color': '#3C4B64'}),
            html.Div([
                html.Div([dcc.Graph(id='pcp-interactive',
                                    hoverData={'points': [{'customdata': ''}]}, config=config)],
                         style={'width': '1000%'}
                         )
            ], style={'overflow': 'scroll', 'border': '1px solid #969696',
                      'border-radius': '5px',
                      'background-color': 'white', 'padding': '0px 20px 20px 10px', 'margin': '0px 20px 20px 20px'}),
            html.Div([html.H2(children="Selected design", id='design-fig-heading', className='widgetTitle'),
                      dcc.Loading(
                          id="loading-design",
                          type="default",
                          children=html.Div(id="loading-output-1"),
                      ),
                      ],
                     style={'padding': '20px 20px 0px 20px', 'color': '#3C4B64'}),
            html.Div([
                dcc.Graph(id='design-fig', figure=blank_fig(),
                          hoverData={'points': [{'customdata': ''}]}, config=config, style={'width': '100%'})
            ], style={'width': '100%', 'padding': '0px 20px 20px 20px',
                      'border': '1px solid #969696', 'border-radius': '5px',
                      'margin': '20px 20px 20px 20px', 'background-color': 'white'}),
            # html.Div([html.H2(children='Power law graph', id='power-law-graph-heading', className='widgetTitle')],
            #          style={'padding': '20px 20px 0px 20px', 'color': '#3C4B64', }),
            # html.Div([
            #     html.Div([dcc.Graph(id='power-law-graph',
            #                         hoverData={'points': [{'customdata': ''}]}, config=config)]),
            # ],
            #     style={'border': '1px solid #969696', 'border-radius': '5px', 'background-color': 'white',
            #            'margin': '0px 20px 20px 20px'}),
            # html.Div([
            #     html.Div([
            #         html.H2(children='Power law evolution', id='power-law-evolution-heading',
            #                 className='widgetTitle')],
            #         style={'width': '80%', 'display': 'inline-block'}
            #     ),
            #     html.Div([
            #         dcc.Input(
            #             id="plaw_evolution_vars",
            #             type="text", placeholder="Var pairs", debounce=True,
            #             inputMode='numeric', value=f"0,1",
            #             style={'width': '100px'}
            #         ),
            #     ], style={'width': '20%', 'display': 'inline-block'})
            # ], style={'padding': '20px 20px 0px 20px', 'color': '#3C4B64'}),
            # html.Div([
            #     html.Div([dcc.Graph(id='power-law-evolution-graph',
            #                         hoverData={'points': [{'customdata': ''}]}, config=config)]),
            # ],
            #     style={'border': '1px solid #969696', 'border-radius': '5px', 'background-color': 'white',
            #            'padding': '0px 20px 20px 20px', 'margin': '0px 20px 0px 20px'})
        ], style={'width': '30%', 'display': 'inline-block', 'float': 'center', 'padding': '20px 20px 20px 20px',
                  'vertical-align': 'top', 'background-color': 'white',
                  'border': '1px solid #969696', 'border-radius': '5px', 'margin': '0px 20px 20px 0px',
                  'overflow': 'scroll', 'height': '83%'}),


        # Shows the rules found by IK-EMO
        html.Div([
            # html.Div([
            #         # html.Div([dcc.Graph(id='design-fig',
            #         #                     hoverData={'points': [{'customdata': ''}]}, config=config)],
            #         #          style={'padding': '20px 20px 20px 20px',  # 'background-color': 'white',
            #         #                 'margin': '0px 0px 20px 0px', 'border-bottom': '1px #EBEDEF solid',
            #         #                 'background-color': 'white'},
            #         #          ),
            #         html.H2(children="Selected design", id='design-fig-heading', className='widgetTitle',
            #                 style={'width': '60%',
            #                        'padding': '20px 20px 20px 20px', 'color': '#3C4B64'}),
            #         dcc.Graph(id='design-fig', figure=blank_fig(),
            #                   hoverData={'points': [{'customdata': ''}]}, config=config)
            #     ], style={'width': '100%', 'display': 'inline-block', 'padding': '0px 20px 20px 20px',
            #               'vertical-align': 'top', 'border': '1px solid #969696', 'border-radius': '5px',
            #               'margin': '0px 20px 20px 0px', 'background-color': 'white', 'height': '100%'}),

        ], style={'width': '34%', 'height': '91%', 'display': 'inline-block'})
    ], style={'width': '100%', 'height': '700px'}),
    html.Div([
        html.Div([


            # Power law rules
            html.H4(children='Power laws (normalized)', id='power-law-rule-list'),
            html.Div([  # Bordered region
                # Rule display settings
                html.Div([
                    html.Div([
                        html.H5(children='Min. rule score', id='maxscore_power_text'),
                        dcc.Input(id="minscore_power", type="number", placeholder="Max. power law score", debounce=True,
                                  inputMode='numeric', value=0),
                    ]),
                    html.Div([
                        html.H5(children='Max. rule evaluation_metric', id='maxerror_power_text'),
                        dcc.Input(id="maxerror_power", type="number", placeholder="Max. power law evaluation_metric", debounce=True,
                                  inputMode='numeric', value=0.01),
                    ]),
                    html.Div([
                        html.H5(children='Min. var. corr.', id='mincorr_power_text'),
                        dcc.Input(id="mincorr_power", type="number", placeholder="Min. power corr", debounce=True,
                                  inputMode='numeric', value=0),
                    ]),
                    # html.Div([
                    #     html.H5(children='Vars per rule', id='varsperrule_power_text'),
                    #     dcc.Input(id="varsperrule_power", type="number", placeholder="Vars per rule", debounce=True,
                    #               inputMode='numeric', value=2, disabled=True),
                    # ]),
                    html.Div([
                        html.Div([
                            dcc.Checklist(
                                id='power-law-select-all',
                                options=[
                                    {'label': 'Select all', 'value': 'select_all'},
                                ],
                                value=[]  # ['NYC', 'MTL']
                            )], style={'display': 'inline-block', 'width': '30%', 'padding': '10px 10px 10px 0px'}
                        ),
                        html.Div([
                            html.Button('Reset', id='power-reset', n_clicks=0)],
                            style={'display': 'inline-block', 'width': '30%', 'padding': '10px 10px 10px 0px'}
                        ),
                        html.H5(children='Mark as rank', id='set_rank'),
                        dcc.Input(id="set_rank_power", type="number", placeholder="Set rank", debounce=True,
                                  inputMode='numeric')
                    ], style={'padding': '10px 0px 10px 0px'})
                ], style={'width': '40%', 'height': '100%', 'display': 'inline-block', 'vertical-align': 'top'}),
                # Rule list
                html.Div([
                    dcc.Checklist(
                        id='power-law-rule-checklist',
                        options=[
                            # {'label': 'New York City', 'value': 'NYC'},
                            # {'label': 'Montréal', 'value': 'MTL'},
                            # {'label': 'San Francisco', 'value': 'SF'}
                        ],
                        value=[]  # ['NYC', 'MTL']
                    )
                ], style={'width': '56%', 'height': '100%', 'display': 'inline-block', 'overflow': 'scroll'})
            ], style={'height': '60%', 'border': '1px solid #969696', 'border-radius': '5px',
                      'background-color': 'white',
                      'padding': '20px 20px 20px 20px'}),

            # Constant rules
            html.H4(children='Constant rules', id='constant-rule-list'),
            html.Div([  # Bordered region
                # Rule display settings
                html.Div([
                    html.Div([
                        html.H5(children='Min. rule score', id='maxscore_constant_text'),
                        dcc.Input(id="minscore_constant", type="number",
                                  placeholder="Max. constant rule score", debounce=True,
                                  inputMode='numeric', value=0),
                    html.Div([
                        html.H5(children='Const. tol.', id='const_tol_text'),
                        dcc.Input(id="const_tol", type="number",
                                  placeholder="Constant rule tol.", debounce=True,
                                  inputMode='numeric', value=0.01),
                    ]),
                    ]),
                    html.Div([
                        html.Div([
                            dcc.Checklist(
                                id='constant-rule-select-all',
                                options=[
                                    {'label': 'Select all', 'value': 'select_all'},
                                ],
                                value=[]  # ['NYC', 'MTL']
                            )], style={'display': 'inline-block', 'width': '30%', 'padding': '10px 10px 10px 0px'}
                        ),
                        html.Div([
                            html.Button('Reset', id='constant-rule-reset', n_clicks=0)],
                            style={'display': 'inline-block', 'width': '30%', 'padding': '10px 10px 10px 0px'}
                        ),
                    ], style={'padding': '10px 0px 10px 0px'})
                ], style={'width': '48%', 'height': '100%', 'display': 'inline-block', 'vertical-align': 'top'}),
                # Rule list
                html.Div([
                    dcc.Checklist(
                        id='constant-rule-checklist',
                        options=[
                            # {'label': 'New York City', 'value': 'NYC'},
                            # {'label': 'Montréal', 'value': 'MTL'},
                            # {'label': 'San Francisco', 'value': 'SF'}
                        ],
                        value=[]  # ['NYC', 'MTL']
                    )
                ], style={'width': '48%', 'height': '100%', 'display': 'inline-block', 'overflow': 'scroll'})
            ], style={'height': '35%', 'border': '1px solid #969696', 'border-radius': '5px',
                      'background-color': 'white',
                      'padding': '20px 20px 20px 20px'}),

            html.H4(children='Inequality rules', id='rule-list'),
            html.Div([  # Bordered region
                # Rule display settings
                html.Div([
                    html.Div([
                        html.H5(children='Min. rule score', id='minscore_ineq_text'),
                        dcc.Input(id="minscore_ineq", type="number", placeholder="Min. ineq. score", debounce=True,
                                  inputMode='numeric', value=0),
                    ], style={'display': 'inline-block'}),
                    html.Div([
                        html.H5(children='Min. var. corr.', id='mincorr_ineq_text'),
                        dcc.Input(id="mincorr_ineq", type="number", placeholder="Min. ineq. corr", debounce=True,
                                  inputMode='numeric', value=0),
                    ]),
                    # html.Div([
                    #     html.H5(children='Vars per rule', id='varsperrule_ineq_text'),
                    #     dcc.Input(id="varsperrule_ineq", type="number", placeholder="Vars per rule", debounce=True,
                    #               inputMode='numeric', value=2, disabled=True),
                    # ]),
                    html.Div([
                        html.Div([
                            dcc.Checklist(
                                id='ineq-select-all',
                                options=[
                                    {'label': 'Select all', 'value': 'select_all'},
                                ],
                                value=[]  # ['NYC', 'MTL']
                            )], style={'display': 'inline-block', 'width': '30%', 'padding': '10px 10px 10px 0px'}
                        ),
                        html.Div([
                            html.Button('Reset', id='ineq-reset', n_clicks=0)],
                            style={'display': 'inline-block', 'width': '30%', 'padding': '10px 10px 10px 0px'}
                        )],
                        style={'padding': '10px 0px 0px 0px'}),
                ], style={'width': '48%', 'height': '100%', 'display': 'inline-block', 'vertical-align': 'top'}),
                # Rule list
                html.Div([
                    dcc.Checklist(
                        id='inequality-rule-checklist',
                        options=[
                            # {'label': 'New York City', 'value': 'NYC'},
                            # {'label': 'Montréal', 'value': 'MTL'},
                            # {'label': 'San Francisco', 'value': 'SF'}
                        ],
                        value=[]  # ['NYC', 'MTL']
                    )
                ], style={'width': '48%', 'height': '100%', 'display': 'inline-block', 'overflow': 'scroll'})
            ], style={'height': '50%', 'border': '1px solid #969696', 'border-radius': '5px',
                      'background-color': 'white', 'padding': '20px 20px 20px 20px'})
        ], style={'height': '100%', 'width': '45%', 'padding': '20px 20px 20px 20px', 'display': 'inline-block',
                  'overflow': 'scroll', 'margin': '20px 20px 20px 20px',
                  'border': '1px solid #969696', 'border-radius': '5px', 'background-color': 'white'}),
        html.Div([
            html.Div([
                html.Span([html.H2(children="Variable relation graph", id='vrg-fig-heading',
                                   style={'width': '45%', 'display': 'inline-block'}, className='widgetTitle'),
                           dcc.Dropdown([], id='var-group-selector', searchable=False,
                                        style={'width': '35%', 'display': 'inline-block'}),
                           # dcc.Input(
                           #     id="vrg_vars",
                           #     type="text", placeholder="Var pairs", debounce=True,
                           #     inputMode='numeric', value=None,
                           #     style={'width': '10%', 'display': 'inline-block'}
                           # ),
                           html.Button('\u2705', id='vrg-include', n_clicks=0, title='Add edge',
                                       style={'padding': '0', 'border': 'none', 'background': 'none',
                                              'margin-left': '20px', 'font-size': '25px'}),
                           html.Button('\u274C', id='vrg-exclude', n_clicks=0, title='Remove edge',
                                       style={'padding': '0', 'border': 'none', 'background': 'none',
                                              'margin-left': '20px', 'font-size': '25px'}),
                           html.Button('\u21BA', id='vrg-reset', n_clicks=0, title='Reset VRG',
                                       style={'padding': '0', 'border': 'none', 'background': 'none',
                                              'margin-left': '20px', 'font-size': '40px'}),
                           ])],
                style={'padding': '20px 0px 0px 20px', 'color': '#3C4B64'}
            ),
            dcc.Graph(id='vrg-fig',
                      hoverData={'points': [{'customdata': ''}]}, config=config),
            html.Div([html.H2(children='Power law graph', id='power-law-graph-heading', className='widgetTitle')],
                                 style={'padding': '20px 20px 0px 20px', 'color': '#3C4B64', }),
            html.Div([
                html.Div([dcc.Graph(id='power-law-graph',
                                    hoverData={'points': [{'customdata': ''}]}, config=config)]),
            ],
                style={'border': '1px solid #969696', 'border-radius': '5px', 'background-color': 'white',
                       'margin': '0px 20px 20px 20px'}),
            html.Div([
                html.Div([
                    html.H2(children='Power law evolution', id='power-law-evolution-heading', className='widgetTitle')],
                    style={'width': '80%', 'display': 'inline-block'}
                ),
                html.Div([
                    dcc.Input(
                        id="plaw_evolution_vars",
                        type="text", placeholder="Var pairs", debounce=True,
                        inputMode='numeric', value=f"0,1",
                        style={'width': '100px'}
                    ),
                ], style={'width': '20%', 'display': 'inline-block'})
            ], style={'padding': '20px 20px 0px 20px', 'color': '#3C4B64'}),
            html.Div([
                html.Div([dcc.Graph(id='power-law-evolution-graph',
                                    hoverData={'points': [{'customdata': ''}]}, config=config)]),
            ],
                style={'border': '1px solid #969696', 'border-radius': '5px', 'background-color': 'white',
                       'padding': '0px 20px 20px 20px', 'margin': '0px 20px 0px 20px'})
        ], style={'width': '45%', 'height': '100%', 'display': 'inline-block', 'padding': '20px 20px 20px 20px',
                  'vertical-align': 'top', 'border': '1px solid #969696', 'border-radius': '5px',
                  'margin': '20px 20px 20px 0px', 'background-color': 'white', 'overflow': 'scroll'})
    ],  style={'height': '700px'}
    ),
    html.Div(id='dummy_rule_rank')
]
app.layout = html.Div(html_layout)


@app.callback(Output('confirm-write', 'displayed'),
              Input('save-data', 'n_clicks'))
def display_confirm(n_clicks):
    ctx = dash.callback_context
    if ctx.triggered:
        # print(ctx.triggered)
        id_which_triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        print(id_which_triggered)
        if id_which_triggered == 'save-data':
            return True
    # if value == 'Danger!!':
    #     return True
    # return False
    return False


@app.callback(Output("hv-evolution-graph", "figure"),
               # Output('play_optim', 'disabled')],
              Input('optim-progress-update-interval', 'n_intervals'),
              # State("optim-loading", "loading_state"),
              # State("play_optim", "n_clicks"),
              )
def update_optim_progress(n_intervals):
    """Display the number of generations completed."""
    # global prob_options
    read_hdf_file()
    hv_list = update_hv_progress(gen_arr)
    hv_fig = get_hv_fig(hv_list)
    print(f"Updating HV figure after {n_intervals} intervals")

    return hv_fig


@app.callback(
    Output(component_id='generation-no', component_property='children'),
    [Input(component_id='cross-filter-gen-slider', component_property='value')]
)
def update_generation_no(selected_gen):
    return f'Generation {selected_gen}'


@app.callback(
    # Output(component_id='cross-filter-gen-slider', component_property='marks'),
    Output(component_id='slider-div', component_property='children'),
    Input(component_id='refresh-data', component_property='n_clicks'),
    State(component_id='cross-filter-gen-slider', component_property='value')

)
def refresh_dashboard(n_clicks, slider_val):
    read_hdf_file()
    slider_steps = get_gen_slider_steps(gen_arr)
    print(slider_steps)
    print(gen_arr)
    slider = dcc.Slider(
        id='cross-filter-gen-slider',
        min=min(gen_arr),
        max=max(gen_arr),
        value=slider_val,
        step=None,
        tooltip={"placement": "bottom", "always_visible": True},
        marks=slider_steps,
        # marks={str(int(gen)): str(int(gen)) for gen in gen_arr}
    )

    return slider


# @app.callback(
#     [Output(component_id='inequality-rule-checklist', component_property='options'),
#      Output(component_id='inequality-rule-checklist', component_property='value')],
#     [Input('cross-filter-gen-slider', 'value'),
#      Input("minscore_ineq", "value"),
#      Input("mincorr_ineq", "value"),
#      Input("varsperrule_ineq", "value"),
#      Input(component_id='objective-space-scatter', component_property='selectedData'),
#      Input(component_id='ineq-reset', component_property='n_clicks')],
#     State(component_id='inequality-rule-checklist', component_property='options')
# )
# def ineq_rule_checklist(selected_gen, minscore_ineq, mincorr_ineq, vars_per_rule, selected_data, n_clicks_power,
#                         checklist):
#     ctx = dash.callback_context
#
#     if ctx.triggered:
#         print("Inequality law triggers")
#         # print(ctx.triggered)
#         id_which_triggered = ctx.triggered[0]['prop_id'].split('.')[0]
#         print(id_which_triggered)
#         if id_which_triggered == 'objective-space-scatter':
#             if selected_data is not None:
#                 print("Inequality law checklist triggered by selected data")
#                 # raise dash.exceptions.PreventUpdate
#         if id_which_triggered == 'ineq-reset':
#             print("Power law reset button pressed")
#             return checklist, []
#     # print("minscore_ineq = ", minscore_ineq)
#     # print("mincorr_power = ", mincorr_ineq)
#
#     if minscore_ineq is not None:
#         innov.min_ineq_rule_significance = float(minscore_ineq)
#     if mincorr_ineq is not None:
#         innov.corr_min_ineq = float(mincorr_ineq)
#
#     # print("innov.min_ineq_rule_significance = ", innov.min_ineq_rule_significance)
#     # print("innov.corr_min_ineq = ", innov.corr_min_ineq)
#
#     all_gen_val = gen_arr
#     nearest_gen_value = int(all_gen_val[np.abs(all_gen_val - selected_gen).argmin()])
#     gen_key = f'gen{nearest_gen_value}'
#     current_gen_data = hf[gen_key]
#
#     obj = np.array(current_gen_data['F'])
#     x = np.array(current_gen_data['X'])
#     rank = np.array(current_gen_data['rank'])
#     f_nd = obj[rank == 0, :]
#     x_nd = x[rank == 0, :]
#     n_var = x_nd.shape[1]
#     xl, xu = hf.attrs['xl'], hf.attrs['xu']
#
#     solution_id = np.array([f"{nearest_gen_value}_{indx}_r{rank[indx]}" for indx in range(obj.shape[0])])
#     solution_id_nd = solution_id[rank == 0]
#
#     data_arr = []
#     # print(selected_data)
#     id_indx_arr = []
#     if selected_data is not None:
#         for data in selected_data['points']:
#             # print(data)
#             data_id = data['customdata']
#             if data_id[-2:] != 'r0':
#                 continue
#             id_indx = np.where(solution_id_nd == data_id)[0][0]
#             data_arr.append(x_nd[id_indx, :].tolist())
#             id_indx_arr.append(id_indx)
#
#         data_arr = np.array(data_arr)
#     else:
#         data_arr = x_nd
#
#     # Inequality rules
#     var_groups, var_group_score, rel_type, corr = innov.learn_inequality_rules(data_arr)
#     var_grp_score_significant = var_group_score[var_group_score >= innov.min_ineq_rule_significance]
#     # print(var_grp_score_significant)
#     var_grp_significant = var_groups[var_group_score >= innov.min_ineq_rule_significance, :]
#     rel_type_significant = rel_type[var_group_score >= innov.min_ineq_rule_significance]
#
#     # print(f"Min score = {innov.min_ineq_rule_significance}")
#     # print(var_grp_score_significant)
#
#     ineq_data = []
#     for i in range(len(var_grp_significant)):
#         indx_min = np.min(var_grp_significant[i, :])
#         indx_max = np.max(var_grp_significant[i, :])
#         if indx_min == indx_max:
#             continue
#         if xl[indx_min] != xl[indx_max] and xu[indx_min] != xu[indx_max]:
#             continue
#         if np.abs(corr[indx_max, indx_min]) < innov.corr_min_ineq:
#             continue
#         # print(var_grp_score_significant[i])
#         istr = f"x{var_grp_significant[i, 0]} <= x{var_grp_significant[i, 1]} ".translate(sub) + \
#                f"(score={np.round(var_grp_score_significant[i], decimals=2)}, " \
#                f"corr={np.round(corr[var_grp_significant[i, 0], var_grp_significant[i, 1]], decimals=2)}, " \
#                f"rtype={rel_type_significant[i]})"
#         ineq_data.append({'label': istr, 'value': var_grp_significant[i]})
#     for i in range(len(ineq_data)):
#         ineq_data[i]['value'] = f"{ineq_data[i]['value'][0]} {ineq_data[i]['value'][1]}"  # str(ineq_data[i]['value'])
#     return ineq_data, []


def get_innovization(current_gen, data_arr, const_tol, rerun=False):
    # Learn power laws and constant vars from selected data.
    innov_file = os.path.join(args.result_path, INNOVIZATION_DIR, f'innov_gen{current_gen}.pkl')
    # For the generations where power laws that were not learned, the learning is performed here and the results are
    # stored in a pickled file. They have a '_post' suffix after the filename to differentiate it.
    innov_file_post = os.path.join(args.result_path, INNOVIZATION_DIR, f'innov_gen{current_gen}_post.pkl')
    if os.path.exists(innov_file) and not rerun:
        with open(innov_file, 'rb') as fp:
            innov = pickle.load(fp)
    elif os.path.exists(innov_file_post) and not rerun:
        with open(innov_file_post, 'rb') as fp:
            innov = pickle.load(fp)
    else:
        var_groups = []
        with h5py.File(hdf_file, 'r') as hf:
            current_gen_data = hf[f'gen{current_gen}']
            for key in current_gen_data.keys():
                if 'var_groups' in key:
                    var_groups.append(np.array(current_gen_data[key]))
        innov = VRGInnovization(n_var=data_arr.shape[1], groups=var_groups, const_tol=const_tol,
                                xl=xl, xu=xu, power_law_normalized=True, agent_names=['power_law_rep_sig_0'])
        innov.learn(data_arr)
        # These innovization rules are learned just now. So write them to the results folder with a '_post' suffix
        with open(innov_file_post, 'wb') as fp:
            pickle.dump(innov, fp)

    return innov


@app.callback(
    [Output(component_id='var-group-selector', component_property='options'),
     Output(component_id='var-group-selector', component_property='value')],
    Input('cross-filter-gen-slider', 'value'),
    State("const_tol", "value")
)
def update_var_group_list(selected_gen, const_tol):
    nearest_gen_value, x, obj, constr, rank, obj_label = get_current_gen_data(selected_gen)
    x_nd = x[rank == 0, :]
    n_var = x_nd.shape[1]
    vrg_innov = get_innovization(nearest_gen_value, x_nd, const_tol)
    var_grp = vrg_innov.groups
    var_group_data = []

    for i in range(len(var_grp)):
        var_group_data.append({'label': f'Group {i + 1}',
                               'value': f'{i}'})

    return var_group_data, ['0']


# @app.callback(
#     Output('cross-filter-gen-slider', 'value'),
#     Input(component_id='playScatter', component_property='n_clicks'),
#     State('cross-filter-gen-slider', 'value'),
# )
# def animate_scatter(nclicks_play_pause, current_gen):
#     ctx = dash.callback_context
#
#     if ctx.triggered:
#         id_which_triggered = ctx.triggered[0]['prop_id'].split('.')[0]
#         print(id_which_triggered)
#         if id_which_triggered == 'playScatter':
#             pass
#     return current_gen


@app.callback(
    [Output(component_id='constant-rule-checklist', component_property='options'),
     Output(component_id='constant-rule-checklist', component_property='value')],
    [Input('cross-filter-gen-slider', 'value'),
     Input("const_tol", "value"),
     Input("minscore_constant", "value"),
     Input(component_id='objective-space-scatter', component_property='selectedData'),
     Input(component_id='constant-rule-reset', component_property='n_clicks'),
     Input("constant-rule-select-all", "value"),
     Input('var-group-selector', 'value')],
    [State(component_id='constant-rule-checklist', component_property='options')]
)
def update_constant_rule_checklist(selected_gen, const_tol, minscore_constant,
                                   selected_data, nclicks_constant, constant_rule_all_selected,
                                   var_grp_selected, constant_rule_options):
    ctx = dash.callback_context

    if ctx.triggered:
        id_which_triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        print(id_which_triggered)
        if id_which_triggered == 'constant-rule-select-all':
            all_or_none = []
            all_or_none = [option["value"] for option in constant_rule_options if constant_rule_all_selected]
            return constant_rule_options, all_or_none
        if id_which_triggered == 'objective-space-scatter':
            if selected_data is not None:
                print("Constant rule checklist triggered by selected data")
                # raise dash.exceptions.PreventUpdate
        if id_which_triggered == 'constant-rule-reset':
            print("Constant rule reset button pressed")
            return constant_rule_options, []

    if var_grp_selected is None:
        v_grp = 0
    else:
        v_grp = int(var_grp_selected[0])
    nearest_gen_value, x, obj, constr, rank, obj_label = get_current_gen_data(selected_gen)
    # f_nd = obj[rank == 0, :]
    x_nd = x[rank == 0, :]
    # n_var = x_nd.shape[1]

    solution_id = np.array([f"{nearest_gen_value}_{indx}_r{rank[indx]}" for indx in range(obj.shape[0])])
    solution_id_nd = solution_id[rank == 0]

    data_arr = []
    # print(selected_data)
    id_indx_arr = []
    if selected_data is not None:
        for data in selected_data['points']:
            # print(data)
            data_id = data['customdata']
            if data_id[-2:] != 'r0':
                continue
            id_indx = np.where(solution_id_nd == data_id)[0][0]
            data_arr.append(x_nd[id_indx, :].tolist())
            id_indx_arr.append(id_indx)

        data_arr = np.array(data_arr)
    else:
        data_arr = x_nd

    # if args.special_flag is not None:
    #     n_var = x.shape[1]
    #     if n_var == 279 or n_var == 86:
    #         n_shape_var = 19
    #     elif n_var == 579 or n_var == 176:
    #         n_shape_var = 39
    #     elif n_var == 879 or n_var == 266:
    #         n_shape_var = 59
    #     else:
    #         return {'data': [], 'layout': None}
    #     for i in range(data_arr.shape[0]):
    #         if n_var == 279 or n_var == 579 or n_var == 879:
    #             symmetry = ()
    #             shape_var = data_arr[i, -n_shape_var:]
    #             shape_var[:n_shape_var // 2 + 1] = np.sort(shape_var[:n_shape_var // 2 + 1])
    #             # shape_var[n_shape_var // 2 + 1:] = np.flip(np.sort(shape_var[n_shape_var // 2 + 1:]))
    #             shape_var[n_shape_var // 2 + 1:] = np.flip(
    #                 shape_var[:n_shape_var // 2 + 1] + 0.001 + np.random.random() * 0.001)[1:]
    #             shape_var[n_shape_var // 2] = (shape_var[n_shape_var // 2] + shape_var[n_shape_var // 2 - 1]) / 2
    #         else:
    #             symmetry = ('xz', 'yz')
    #             shape_var = data_arr[i, -(n_shape_var // 2 + 1):]
    #             shape_var = np.sort(shape_var)
    #         data_arr[i, -n_shape_var:] = shape_var

    innov = get_innovization(nearest_gen_value, data_arr, const_tol, rerun=True)
    data_arr_normalized = innov.normalize_data(data_arr)

    const_var_list = np.where(innov.relation[0].const_var_flag == 1)[0]
    const_c = innov.relation[0].c
    constant_rule_data = []
    # Show rules of type x = constant.
    print("Var grp = ", innov.groups[v_grp])
    print("const_var_list = ", const_var_list)
    # for i in const_var_list:
    for i in range(len(innov.groups[v_grp]) - 1):
        var_i = innov.groups[v_grp][i]
        if var_i in const_var_list:
            # TODO: Convert cstr creation to a function for re-usability.
            # std_x = np.std(data_arr, axis=0)
            # if i in const_var_list:
            diff_i = np.abs(data_arr_normalized[:, var_i] - const_c[var_i])
            score_const = len(np.where(diff_i <= innov.relation[0].const_tol)[0]) / data_arr.shape[0]
            const_c_original = innov.xl[var_i] + (const_c[var_i] - innov.normalize_to_range[0]) / (
                        innov.normalize_to_range[1] - innov.normalize_to_range[0]) * (innov.xu[var_i] - innov.xl[var_i])
            # cstr = f"x\u0302{i} = ".translate(sub) \
            #        + f"{np.round(const_c[i], decimals=2)} " \
            #          f"(score = {np.round(score_const, decimals=2)})"
            cstr = f"x{var_i} = ".translate(sub) \
                   + f"{np.round(const_c_original, decimals=2)} " \
                     f"(score = {np.round(score_const, decimals=2)})"
            eq_law = [var_i, const_c[var_i]]
            if score_const >= minscore_constant:
                constant_rule_data.append({'label': cstr, 'value': eq_law})

    for i in range(len(constant_rule_data)):
        constant_rule_data[i]['value'] = str(constant_rule_data[i]['value'])

    return constant_rule_data, []


@app.callback(
    [Output(component_id='power-law-rule-checklist', component_property='options'),
     Output(component_id='power-law-rule-checklist', component_property='value')],
    [Input('cross-filter-gen-slider', 'value'),
     Input("maxerror_power", "value"),
     Input("minscore_power", "value"),
     Input("mincorr_power", "value"),
     # Input("varsperrule_power", "value"),
     Input(component_id='objective-space-scatter', component_property='selectedData'),
     Input(component_id='power-reset', component_property='n_clicks'),
     Input("power-law-select-all", "value"),
     Input('var-group-selector', 'value')],
    [State(component_id='power-law-rule-checklist', component_property='options'),
     State("maxerror_power", "value"),
     State("const_tol", "value")]
)
def update_power_law_rule_checklist(selected_gen, maxerror_power, minscore_power, mincorr_power,
                                    selected_data, nclicks_power, power_law_all_selected,
                                    var_grp_selected,
                                    power_law_options, power_law_max_error, const_tol):
    ctx = dash.callback_context

    if ctx.triggered:
        print("Power law triggers")
        # print(ctx.triggered)
        id_which_triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        print(id_which_triggered)
        if id_which_triggered == 'power-law-select-all':
            all_or_none = []
            all_or_none = [option["value"] for option in power_law_options if power_law_all_selected]
            return power_law_options, all_or_none
        if id_which_triggered == 'objective-space-scatter':
            if selected_data is not None:
                print("Power law checklist triggered by selected data")
                # raise dash.exceptions.PreventUpdate
        if id_which_triggered == 'power-reset':
            print("Power law reset button pressed")
            return power_law_options, []

    if var_grp_selected is None:
        v_grp = 0
    else:
        v_grp = int(var_grp_selected[0])
    nearest_gen_value, x, obj, constr, rank, obj_label = get_current_gen_data(selected_gen)

    # f_nd = obj[rank == 0, :]
    x_nd = x[rank == 0, :]
    # n_var = x_nd.shape[1]

    solution_id = np.array([f"{nearest_gen_value}_{indx}_r{rank[indx]}" for indx in range(obj.shape[0])])
    solution_id_nd = solution_id[rank == 0]

    data_arr = []
    # print(selected_data)
    id_indx_arr = []
    if selected_data is not None:
        for data in selected_data['points']:
            # print(data)
            data_id = data['customdata']
            if data_id[-2:] != 'r0':
                continue
            id_indx = np.where(solution_id_nd == data_id)[0][0]
            data_arr.append(x_nd[id_indx, :].tolist())
            id_indx_arr.append(id_indx)

        data_arr = np.array(data_arr)
    else:
        data_arr = x_nd

    # if args.special_flag is not None:
    #     n_var = x.shape[1]
    #     if n_var == 279 or n_var == 86:
    #         n_shape_var = 19
    #     elif n_var == 579 or n_var == 176:
    #         n_shape_var = 39
    #     elif n_var == 879 or n_var == 266:
    #         n_shape_var = 59
    #     else:
    #         return {'data': [], 'layout': None}
    #     for i in range(data_arr.shape[0]):
    #         if n_var == 279 or n_var == 579 or n_var == 879:
    #             symmetry = ()
    #             shape_var = data_arr[i, -n_shape_var:]
    #             shape_var[:n_shape_var // 2 + 1] = np.sort(shape_var[:n_shape_var // 2 + 1])
    #             # shape_var[n_shape_var // 2 + 1:] = np.flip(np.sort(shape_var[n_shape_var // 2 + 1:]))
    #             shape_var[n_shape_var // 2 + 1:] = np.flip(
    #                 shape_var[:n_shape_var // 2 + 1] + 0.001 + np.random.random() * 0.001)[1:]
    #             shape_var[n_shape_var // 2] = (shape_var[n_shape_var // 2] + shape_var[n_shape_var // 2 - 1]) / 2
    #         else:
    #             symmetry = ('xz', 'yz')
    #             shape_var = data_arr[i, -(n_shape_var // 2 + 1):]
    #             shape_var = np.sort(shape_var)
    #         data_arr[i, -n_shape_var:] = shape_var

    innov = get_innovization(nearest_gen_value, data_arr, const_tol)

    b_arr, c_arr = innov.relation[1].b, innov.relation[1].c
    power_law_error = innov.relation[1].evaluation_metric
    const_var_list = np.where(innov.relation[0].const_var_flag == 1)[0]
    power_law_data = []
    print("Var grp = ", innov.groups[v_grp])
    # TODO: Following lists will be used for sorting the listed power laws
    power_law_score_list = []
    power_law_error_list = []
    power_law_corr_list = []

    # For every var pair in the currently selected group
    for i in range(len(innov.groups[v_grp]) - 1):
        var_i = innov.groups[v_grp][i]
        if var_i in const_var_list:
            continue
        for j in range(i + 1, len(innov.groups[v_grp])):
            var_j = innov.groups[v_grp][j]
            if var_j in const_var_list:
                continue
            # Show rules for normal power laws.
            if var_i not in const_var_list and var_j not in const_var_list:
                if power_law_error[var_i, var_j] > maxerror_power:
                    continue
                if np.abs(innov.correlation[var_i, var_j]) < mincorr_power:
                    continue

                # score = 0
                # x_nd_normalized = normalize_x(data_arr)
                power_law = [var_i, var_j, b_arr[var_i, var_j], c_arr[var_i, var_j]]

                rule_compliance_id, rule_compliance_id_power, rule_compliance_id_ineq \
                    = get_rule_compliance(x_nd=x_nd, var_grp=None, curr_gen=nearest_gen_value, power_law=[power_law],
                                          power_law_max_error=power_law_max_error, const_tol=const_tol)
                score = len(rule_compliance_id) / data_arr.shape[0]

                if score < minscore_power:
                    continue

                pstr = f"x\u0302{var_i} * x\u0302{var_j}".translate(sub) \
                       + f"{np.round(b_arr[var_i, var_j], decimals=2)}".translate(sup) \
                       + f" = {np.round(c_arr[var_i, var_j], decimals=2)}" \
                       + f" (score = {np.round(score, decimals=2)}," \
                       + " {} = {:.1e},".format(innov.relation[1].evaluation_metric_name, power_law_error[var_i, var_j]) \
                       + f" corr = {np.round(innov.correlation[var_i, var_j], decimals=2)}" \
                       + f" )"
                power_law_data.append({'label': pstr,
                                       'value': power_law})
                power_law_score_list.append(np.round(score, decimals=2))
                power_law_error_list.append(power_law_error[var_i, var_j])
                power_law_corr_list.append(np.round(innov.correlation[var_i, var_j], decimals=2))

    for i in range(len(power_law_data)):
        power_law_data[i]['value'] = str(power_law_data[i]['value'])

    return power_law_data, []


def get_rule_compliance(x_nd, var_grp, curr_gen, power_law, power_law_max_error, const_tol):
    rule_compliance_id_ineq = np.array([])
    if var_grp is not None:
        for grp in var_grp:
            id_list = np.where(x_nd[:, grp[0]] <= x_nd[:, grp[1]])[0]
            if len(rule_compliance_id_ineq) == 0:
                rule_compliance_id_ineq = id_list
            else:
                rule_compliance_id_ineq = np.intersect1d(rule_compliance_id_ineq, id_list)

    # if power_law.ndim == 1:
    #     power_law = power_law.reshape([1, -1])

    vrg_innov = get_innovization(curr_gen, x_nd, const_tol)
    x_nd_normalized = vrg_innov.normalize_data(x_nd)

    n_var = x_nd.shape[1]
    rule_compliance_id_power = []
    for sol_indx, x_sol in enumerate(x_nd_normalized):
        # power_law_compliance = np.zeros([n_var, n_var])
        compliance_flag = True
        for law in power_law:
            i, j = law[:2]
            i = int(i)
            j = int(j)
            compliance = vrg_innov.relation[1].check_compliance(x_test=x_sol,
                                                                var_pair=np.array([[i, j]]),
                                                                error_threshold=power_law_max_error,
                                                                ignore_vars=ignore_vars)
            if compliance[i, j] != 1:
                compliance_flag = False
        if len(power_law) > 0 and compliance_flag:
            rule_compliance_id_power.append(sol_indx)

    if len(rule_compliance_id_ineq) == 0:
        rule_compliance_id = rule_compliance_id_power
    elif len(rule_compliance_id_power) == 0:
        rule_compliance_id = rule_compliance_id_ineq
    else:
        rule_compliance_id = np.intersect1d(rule_compliance_id_ineq, rule_compliance_id_power)
    # print(f"No. of pop members following selected rule(s) = {len(rule_compliance_id)}")

    return rule_compliance_id, rule_compliance_id_power, rule_compliance_id_ineq

@app.callback(
    Output('objective-space-scatter', 'figure'),
    [Input('cross-filter-gen-slider', 'value'),
     Input('inequality-rule-checklist', 'value'),
     Input('power-law-rule-checklist', 'value'),
     Input(component_id='objective-space-scatter', component_property='selectedData'),
     Input(component_id='objective-space-scatter', component_property='clickData')],
    [State("maxerror_power", "value"),
     State("const_tol", "value")]
)
def update_objective_space_scatter_graph(selected_gen, var_grp_str, power_law_str, selected_data, click_data,
                                         power_law_max_error, const_tol):
    ctx = dash.callback_context
    power_law = copy.deepcopy(power_law_str)
    for i, law in enumerate(power_law):
        power_law[i] = convert_checklist_str_to_list(law)
        for j in range(len(power_law[i])):
            power_law[i][j] = float(power_law[i][j])
    var_grp = copy.deepcopy(var_grp_str)
    for i, grp in enumerate(var_grp):
        var_grp[i] = convert_checklist_str_to_list(var_grp[i])[0].split(' ')
        for j in range(len(var_grp[i])):
            var_grp[i][j] = int(var_grp[i][j])
    layout_update = True
    click_flag = True
    if ctx.triggered:
        trigger_prop_id = ctx.triggered[0]['prop_id'].split('.')
        id_which_triggered = trigger_prop_id[0]
        prop_which_triggered = trigger_prop_id[1]
        print(id_which_triggered)
        print(prop_which_triggered)
        if id_which_triggered == 'objective-space-scatter':
            if prop_which_triggered == 'clickData':
                click_flag = True
            elif selected_data is not None:
                print("Exception raised")
                raise dash.exceptions.PreventUpdate
        elif id_which_triggered == 'power-law-rule-checklist':
            print(power_law)
            print(selected_data)
            # layout_update = False
            if selected_data is not None and len(power_law) == 0:
                print("Power law trigger obj graph")
                raise dash.exceptions.PreventUpdate
        elif id_which_triggered == 'inequality-rule-checklist':
            # layout_update = False
            if selected_data is not None and len(var_grp) == 0:
                print("Inequality trigger obj graph")
                raise dash.exceptions.PreventUpdate

    nearest_gen_value, x, obj, constr, rank, obj_label = get_current_gen_data(selected_gen)
    f_nd = obj[rank == 0, :]
    x_nd = x[rank == 0, :]
    # x_dominated = x[rank > 0, :]
    f_dominated = obj[rank > 0, :]
    n_obj = obj.shape[1]

    solution_id = np.array([f"{nearest_gen_value}_{indx}_r{rank[indx]}" for indx in range(obj.shape[0])])
    solution_id_nd = solution_id[rank == 0]

    rule_compliance_id, rule_compliance_id_power, rule_compliance_id_ineq \
        = get_rule_compliance(x_nd=x_nd, var_grp=var_grp, curr_gen=nearest_gen_value, power_law=power_law,
                              power_law_max_error=power_law_max_error, const_tol=const_tol)

    f_nd_unselected = np.copy(f_nd)
    solution_id_nd_unselected = np.copy(solution_id_nd)
    if len(rule_compliance_id) > 0:
        del_row_indx = []
        f_rule = obj[rule_compliance_id, :]
        solution_id_rule = solution_id[rule_compliance_id]
        for id_indx, id in enumerate(solution_id_nd_unselected):
            if id in solution_id_rule:
                del_row_indx.append(id_indx)

        f_nd_unselected = np.delete(f_nd_unselected, del_row_indx, axis=0)
        solution_id_nd_unselected = np.delete(solution_id_nd_unselected, del_row_indx, axis=0)

    # TODO: Make 3d graph plot possible
    return_data = {'data': []}

    if n_obj == 2:
        return_data['data'] += [
            go.Scatter(
                x=f_dominated[:, 0],
                y=f_dominated[:, 1],
                customdata=solution_id,
                mode='markers',
                name='Dominated',
                marker={
                    'size': 10,
                    'opacity': 0.5,
                    'line': {'width': 0.5, 'color': 'white'}  # TODO: Why is this set to white?
                },
            ),
            go.Scatter(
                x=f_nd_unselected[:, 0],
                y=f_nd_unselected[:, 1],
                customdata=solution_id_nd_unselected,
                mode='markers',
                name='Non-dominated',
                marker={
                    'size': 10,
                    'opacity': 0.8,
                    'color': 'OrangeRed',
                    'line': {'width': 0.5, 'color': 'white'}
                },
            ),
        ]
        # TODO: Selecting a xi = c type rule does not highlight points on the scatter plot
        if len(rule_compliance_id) > 0:
            f_rule = obj[rule_compliance_id, :]
            solution_id_rule = solution_id[rule_compliance_id]
            return_data['data'].append(
                go.Scatter(
                    x=f_rule[:, 0],
                    y=f_rule[:, 1],
                    customdata=solution_id_rule,
                    mode='markers',
                    name='Rule compliance',
                    marker={
                        'size': 10,
                        'opacity': 0.6,
                        'color': 'Green',
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    # legendrank=2000,
                ))

        if click_flag and click_data is not None:
            return_data['data'] += [
                go.Scatter(
                    x=[click_data['points'][0]['x']],
                    y=[click_data['points'][0]['y']],
                    # customdata=solution_id_nd_unselected,
                    mode='markers',
                    name='Clicked data',
                    marker={
                        'size': 10,
                        'opacity': 0.8,
                        'symbol': 'circle-open',
                        'line': {'width': 4, 'color': 'blue'}
                    },
                    showlegend=False,
                )
            ]

    xaxis = {
        'title': obj_label[0],
        'titlefont': {'size': 18},
        'tickfont': {'size': 16},
        # 'showline': True,
        'linecolor': 'black',
        'zeroline': False,
        'mirror': True,
        'type': 'linear',
        'autorange': True,
        'automargin': True,
        # 'rangemode': 'tozero'
    }
    yaxis = {
        'title': obj_label[1],
        'titlefont': {'size': 18},
        'tickfont': {'size': 16},
        'tickprefix': "   ",
        # 'showline': True,
        'linecolor': 'black',
        'zeroline': False,
        'mirror': True,
        'type': 'linear',
        'autorange': True,
        'automargin': True,
        # 'rangemode': 'tozero'
    }
    # if layout_update:
    #     xaxis['autorange'] = True
    #     yaxis['autorange'] = True

    return_data['layout'] = go.Layout(
        xaxis=xaxis,
        yaxis=yaxis,
        margin={'l': 50, 'b': 30, 't': 10, 'r': 0},
        height=400,
        legend=dict(orientation="v",
                    x=0.55, y=0.95, xanchor='left', font={'size': 14}, bordercolor="Black", borderwidth=1),
        hovermode='closest',
    )

    return return_data


def convert_checklist_str_to_list(power_law):
    return power_law.strip('][').split(', ')


@app.callback(
    Output('power-law-evolution-graph', 'figure'),
    Input('plaw_evolution_vars', 'value'),
    [State('cross-filter-gen-slider', 'value'),
     State("const_tol", "value")]
)
def update_power_law_evolution_plot(plaw_var, current_gen, const_tol):
    plaw_str = plaw_var.split(',')
    i, j = int(plaw_str[0]), int(plaw_str[1])
    return_data = {'data': []}
    gen_count = 0
    for gen in gen_arr:
        if (gen + 1) % 20 == 0 or gen == current_gen:
            gen_count += 1

    alpha = np.linspace(0, 1, num=gen_count)
    cntr = 0
    for indx, gen in enumerate(gen_arr):
        if (gen + 1) % 20 == 0 or gen == current_gen:
            nearest_gen_value, x, obj, constr, rank, obj_label = get_current_gen_data(gen)
            training_data_gen_indx = nearest_gen_value - 1  # Training data is the previous element in gen_arr

            f_nd = obj[rank == 0, :]
            x_nd = x[rank == 0, :]
            n_var = x_nd.shape[1]
            vrg_innov = get_innovization(nearest_gen_value, x_nd, const_tol)

            b, c = vrg_innov.relation[1].b[i, j], vrg_innov.relation[1].c[i, j]
            ll_i, ul_i = vrg_innov.normalize_to_range[0], vrg_innov.normalize_to_range[1]
            ll_j, ul_j = vrg_innov.normalize_to_range[0], vrg_innov.normalize_to_range[1]

            xj = np.linspace(ll_j, ul_j, int(100 * (ul_j - ll_j)))
            xi = c / (xj ** b)
            xj = xj[(xi >= ll_i) & (xi <= ul_i)]
            xi = xi[(xi >= ll_i) & (xi <= ul_i)]

            return_data['data'] += \
                [
                    go.Scatter(
                        x=xi,
                        y=xj,
                        mode='lines',
                        # name=f"x\u0302{i} * x\u0302{j}".translate(sub) + f"{np.round(b, decimals=2)}".translate(sup) +
                        #      f" = {np.round(c, decimals=2)}",
                        name=f"Gen {gen}",
                        marker={
                            'size': 10,
                            'line': {'width': 0.5, 'color': 'white'},
                            # 'opacity': alpha[cntr],
                            # 'color': 'Red',
                        },
                        showlegend=True,
                    )
                ]
            cntr += 1

    wl, wu = 0.95, 1.05
    return_data['layout'] = go.Layout(
        xaxis={
            'title': f'x\u0302{i}'.translate(sub),
            'titlefont': {'size': 18},
            'tickfont': {'size': 16},
            # 'showline': True,
            'linecolor': 'black',
            'zeroline': False,
            'mirror': True,
            'type': 'linear',
            # 'autorange': True,
            'automargin': True,
            # 'rangemode': 'tozero',
            'range': [vrg_innov.normalize_to_range[0] * wl, vrg_innov.normalize_to_range[1] * wu],
        },
        yaxis={
            'title': f'x\u0302{j}'.translate(sub),
            'titlefont': {'size': 18},
            'tickfont': {'size': 16},
            'tickprefix': "   ",
            # 'showline': True,
            'linecolor': 'black',
            'zeroline': False,
            'mirror': True,
            'type': 'linear',
            # 'autorange': True,
            'automargin': True,
            # 'rangemode': 'tozero',
            'range': [vrg_innov.normalize_to_range[0] * wl, vrg_innov.normalize_to_range[1] * wu],
        },
        margin={'l': 50, 'b': 30, 't': 10, 'r': 0},
        height=400,
        legend=dict(orientation="v",
                    x=0.75, y=0.95, xanchor='left', font={'size': 14}, bordercolor="Black", borderwidth=1),
        hovermode='closest',
    )

    return return_data


@app.callback(
    Output(component_id='pause-continue-optimization', component_property='children'),
    Input(component_id='pause-continue-optimization', component_property='n_clicks'),
    State(component_id='pause-continue-optimization', component_property='children'),
)
def toggle_pause_button(pause_click, title):
    ctx = dash.callback_context
    pause_file_path = os.path.join(args.result_path, '.pauserun')
    if ctx.triggered:
        # print(ctx.triggered)
        id_which_triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        print(id_which_triggered)
        if id_which_triggered == 'pause-continue-optimization':
            # When we want to pause the optimization
            if title == pause_icon:
                with open(pause_file_path, 'w') as _:
                    pass
                return play_icon
            # When we want to resume the optimization
            elif title == play_icon:
                if os.path.exists(pause_file_path):
                    os.remove(pause_file_path)
                return pause_icon
            else:
                print("Unknown title")
                return title
    return title


@app.callback(
    Output('vrg-fig', 'figure'),
    [Input('cross-filter-gen-slider', 'value'),
     Input(component_id='vrg-include', component_property='n_clicks'),
     Input(component_id='vrg-exclude', component_property='n_clicks'),
     Input(component_id='vrg-reset', component_property='n_clicks'),
     Input('var-group-selector', 'value')],
    [State(component_id='objective-space-scatter', component_property='selectedData'),
     State('power-law-rule-checklist', 'value'),
     # State('vrg_vars', 'value'),
     State("maxerror_power", "value"),
     State("const_tol", "value")]
)
def update_vrg_plot(selected_gen, include_click, exclude_click, reset_click, var_grp_selected,
                    selected_data, power_law, power_law_max_error, const_tol):
    """Displays the variable relation graph."""
    nearest_gen_value, x, obj, constr, rank, obj_label = get_current_gen_data(selected_gen)

    f_nd = obj[rank == 0, :]
    x_nd = x[rank == 0, :]
    n_var = x_nd.shape[1]

    solution_id = np.array([f"{nearest_gen_value}_{indx}_r{rank[indx]}" for indx in range(obj.shape[0])])
    solution_id_nd = solution_id[rank == 0]

    data_arr = []
    # print(selected_data)
    id_indx_arr = []
    if selected_data is not None:
        for data in selected_data['points']:
            # print(data)
            data_id = data['customdata']
            if data_id[-2:] != 'r0':
                continue
            id_indx = np.where(solution_id_nd == data_id)[0][0]
            data_arr.append(x_nd[id_indx, :].tolist())
            id_indx_arr.append(id_indx)

        data_arr = np.array(data_arr)
    else:
        data_arr = x_nd

    innov = get_innovization(nearest_gen_value, data_arr, const_tol)
    if var_grp_selected is None:
        v_grp = 0
    else:
        v_grp = int(var_grp_selected[0])
    print(f"Selected var grp = {v_grp}")
    nx_graph = copy.deepcopy(innov.vrg[v_grp].graph)  # Get the networkx plot
    node_pos = nx.circular_layout(nx_graph)

    ctx = dash.callback_context

    selected_vars = []
    power_law_dict = {}
    for indx, law_str in enumerate(power_law):
        law = convert_checklist_str_to_list(law_str)
        if len(law) == 4:
            i, j, b, c = law
            i = int(i)
            j = int(j)
            b = float(b)
            c = float(c)
            selected_vars.append([i, j])

            power_law = [i, j, b, c]

            rule_compliance_id, rule_compliance_id_power, rule_compliance_id_ineq \
                = get_rule_compliance(x_nd=x_nd, var_grp=None, curr_gen=nearest_gen_value, power_law=[power_law],
                                      power_law_max_error=power_law_max_error, const_tol=const_tol)
            # score = len(rule_compliance_id) / data_arr.shape[0]
            # power_law_dict[(i, j)] = score

    # VRG edges to be added/removed by the user
    vrg_vars_to_exclude = []
    vrg_vars_to_include = list(nx_graph.nodes())
    id_which_triggered = None
    if ctx.triggered:
        # print(ctx.triggered)
        id_which_triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        print(id_which_triggered)
        if id_which_triggered == 'vrg-exclude':
            print("VRG Exclude")
            # vrg_vars_to_exclude = [int(v) for v in vrg_vars_str.split(',')]
            vrg_vars_to_exclude = copy.copy(selected_vars)
            print("VRG vars to exclude = ", vrg_vars_to_exclude)
        elif id_which_triggered == 'vrg-include':
            print("VRG Include")
            # vrg_vars_to_include = [int(v) for v in vrg_vars_str.split(',')]
            vrg_vars_to_include = copy.copy(selected_vars)
            print("VRG vars to include = ", vrg_vars_to_include)

    # Create graph edges
    edge_x = []
    edge_y = []
    edge_trace_list = []  # For varying the edge thickness according to rule score
    score_list = []
    all_edges = list(nx_graph.edges())
    # change_flag = False
    for edge in all_edges:
        # If edge is excluded by user
        if id_which_triggered == 'vrg-exclude':
            if [edge[0], edge[1]] in vrg_vars_to_exclude or [edge[1], edge[0]] in vrg_vars_to_exclude:
                nx_graph.remove_edge(edge[0], edge[1])
                # change_flag = True
                continue
        # If edge is included by user
        elif id_which_triggered == 'vrg-include':
            if [edge[0], edge[1]] not in vrg_vars_to_include and [edge[1], edge[0]] not in vrg_vars_to_include:
                nx_graph.remove_edge(edge[0], edge[1])
                # change_flag = True
                continue

        power_law = [edge[0], edge[1], innov.relation[1].b[edge[0], edge[1]], innov.relation[1].c[edge[0], edge[1]]]

        rule_compliance_id, rule_compliance_id_power, rule_compliance_id_ineq \
            = get_rule_compliance(x_nd=x_nd, var_grp=None, curr_gen=nearest_gen_value, power_law=[power_law],
                                  power_law_max_error=power_law_max_error, const_tol=const_tol)
        score = len(rule_compliance_id) / data_arr.shape[0]
        score_list.append(score)
        # x0, y0 = G.nodes[edge[0]]['pos']
        # x1, y1 = G.nodes[edge[1]]['pos']
        x0, y0 = node_pos[edge[0]]
        x1, y1 = node_pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

        edge_width = 0.5 + score*(5 - 0.5)
        print(edge[0], edge[1], score, edge_width)
        edge_trace_list.append(
            go.Scatter(x=[x0, x1], y=[y0, y1], hovertemplate=f"Score = {score}",
                       line=dict(width=edge_width, color='#888'), mode='lines'))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    all_node_list = list(nx_graph.nodes())
    for node in all_node_list:
        if nx_graph.degree[node] == 0:
            nx_graph.remove_node(node)
            continue
        # x, y = G.nodes[node]['pos']
        x, y = node_pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        text=[str(node_str) for node_str in list(nx_graph.nodes())],
        textfont={'size': 30, 'color': 'white'},
        textposition="middle center",
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            # colorscale='YlGnBu',
            colorscale='bluered',
            reversescale=False,
            color=[],
            size=60,
            colorbar=dict(
                dtick=1,
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            # line_width=2
        ))

    # Color node points
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(nx_graph.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: ' + str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.hovertext = node_text

    # Create network graphs
    fig = go.Figure(
        # data=[edge_trace, node_trace],
                    data=edge_trace_list + [node_trace],
                    layout=go.Layout(
                        # title='<br>Network graph made with Python',
                        # title=go.layout.Title(text='Variable relation graph (VRG)', font={'size': 16}),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        # annotations=[dict(
                        #     text="Python code: "
                        #          "<a href='https://plotly.com/ipython-notebooks/network-graphs/'> "
                        #          "https://plotly.com/ipython-notebooks/network-graphs/</a>",
                        #     showarrow=False,
                        #     xref="paper", yref="paper",
                        #     x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    # if change_flag:
    #     innov.vrg[v_grp].graph = nx_graph'

    return fig


@app.callback(
    Output('power-law-graph', 'figure'),
    [Input('power-law-rule-checklist', 'value'),
     Input('constant-rule-checklist', 'value')],
    [State('cross-filter-gen-slider', 'value'),
     State("const_tol", "value")]
)
def update_power_law_plot(power_law, constant_rule, selected_gen, const_tol):
    return_data = {'data': []}
    plaw_evolution_plot = {'data': []}

    nearest_gen_value, x, obj, constr, rank, obj_label = get_current_gen_data(selected_gen)
    training_data_gen_indx = nearest_gen_value - 1  # Training data is the previous element in gen_arr

    f_nd = obj[rank == 0, :]
    x_nd = x[rank == 0, :]
    n_var = x_nd.shape[1]
    vrg_innov = get_innovization(nearest_gen_value, x_nd, const_tol)
    x_nd_normalized = vrg_innov.normalize_data(x_nd)
    legendgroup = 1
    plaw_evolution_var_pair = None
    for indx, law_str in enumerate(power_law):
        law = convert_checklist_str_to_list(law_str)
        if len(law) == 4:
            i, j, b, c = law
            i = int(i)
            j = int(j)
            b = float(b)
            c = float(c)

            ll_i, ul_i = vrg_innov.normalize_to_range[0], vrg_innov.normalize_to_range[1]
            ll_j, ul_j = vrg_innov.normalize_to_range[0], vrg_innov.normalize_to_range[1]

            xj = np.linspace(ll_j, ul_j, int(100 * (ul_j - ll_j)))
            xi = c / (xj ** b)
            xj = xj[(xi >= ll_i) & (xi <= ul_i)]
            xi = xi[(xi >= ll_i) & (xi <= ul_i)]

            return_data['data'] += \
                [
                    go.Scatter(
                        x=xi,
                        y=xj,
                        mode='lines',
                        name=f"x\u0302{i} * x\u0302{j}".translate(sub) + f"{np.round(b, decimals=2)}".translate(sup) +
                             f" = {np.round(c, decimals=2)}",
                        marker={
                            'size': 10,
                            'opacity': 0.5,
                            'line': {'width': 0.5, 'color': 'white'}
                        },
                        showlegend=True,
                        # legendgroup=f'group{legendgroup}'
                    ),
                    go.Scatter(
                        x=x_nd_normalized[:, i],
                        y=x_nd_normalized[:, j],
                        mode='markers',
                        # name=f"Actual x\u0302{i}, x\u0302{j}".translate(sub),
                        name=f"Offspring ND set".translate(sub),
                        marker={
                            # 'size': 10,
                            'opacity': 0.5,
                            'symbol': 'x'
                        },
                        showlegend=False,
                        # legendgroup=f'group{legendgroup}'
                    ),
                ]
            # plaw_evolution_plot['data'] = copy.copy(return_data['data'][-2:])
            plaw_evolution_var_pair = [i, j, np.round(b, decimals=2), np.round(c, decimals=2)]

            # if training_data_gen_indx >= 0:
            #     training_data_gen = int(all_gen_val[training_data_gen_indx])
            #     training_data_gen_key = f'gen{training_data_gen}'
            #     training_data_gen_data = hf[training_data_gen_key]
            #     x_train_rank = np.array(training_data_gen_data['rank'])
            #     x_train = np.array(training_data_gen_data['X'])[x_train_rank == 0]
            #     # print(x_train)
            #     x_train_normalized = innov.normalize_to_range[0] + (x_train - xl) / (xu - xl) * (
            #             innov.normalize_to_range[1] - innov.normalize_to_range[0])
            #     if len(x_train) > 0:
            #         return_data['data'] += \
            #             [
            #                 go.Scatter(
            #                     x=x_train_normalized[:, i],
            #                     y=x_train_normalized[:, j],
            #                     mode='markers',
            #                     name=f"Training data",
            #                     marker={
            #                         # 'size': 10,
            #                         'opacity': 0.5,
            #                         'symbol': 'x'
            #                     },
            #                     showlegend=True,
            #                     # legendgroup=f'group{legendgroup}'
            #                 )
            #         ]
            #     else:
            #         print("Training data not found.")
        # elif len(law) == 2:

    for indx, law_str in enumerate(constant_rule):
        law = convert_checklist_str_to_list(law_str)
        j, mean_xj = law
        j = int(j)
        mean_xj = float(mean_xj)
        ll_i, ul_i = vrg_innov.normalize_to_range[0], vrg_innov.normalize_to_range[1]
        xi = np.linspace(ll_i, ul_i, int(100 * (ul_i - ll_i)))
        # xj = ((innov.normalize_to_range[0]
        #       + (mean_xj - xl[j])/(xu[j] - xl[j])*(innov.normalize_to_range[1] - innov.normalize_to_range[0]))
        #       * np.ones_like(xi))
        xj = mean_xj * np.ones_like(xi)
        return_data['data'] += \
            [
                go.Scatter(
                    x=xi,
                    y=xj,
                    mode='lines',
                    name=f"x\u0302{j}".translate(sub) + f"= {np.round(mean_xj, decimals=2)}",
                    marker={
                        'size': 10,
                        'opacity': 0.5,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    showlegend=True
                ),
            ]

        # else:
        #     warnings.warn("Incorrect length of power law list.")
    # Plot xi = xj line
    xi = np.linspace(vrg_innov.normalize_to_range[0], vrg_innov.normalize_to_range[1], 100)
    return_data['data'] += \
        [
            go.Scatter(
                x=xi,
                y=xi,
                mode='lines',
                name='x\u0302i'.translate(sub_ij) + ' = x\u0302j'.translate(sub_ij),
                line=dict(color="gray", dash='dash'),
                marker={
                    'size': 10,
                    'opacity': 0.5,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                showlegend=True
            ),
        ]
    wl, wu = 0.95, 1.05
    return_data['layout'] = go.Layout(
        xaxis={
            'title': 'x\u0302i'.translate(sub_ij),
            'titlefont': {'size': 20},
            'tickfont': {'size': 18},
            # 'showline': True,
            'linecolor': 'black',
            'zeroline': False,
            'mirror': True,
            'type': 'linear',
            # 'autorange': True,
            # 'automargin': True,
            # 'rangemode': 'tozero',
            'range': [vrg_innov.normalize_to_range[0] * wl, vrg_innov.normalize_to_range[1] * wu],
        },
        yaxis={
            'title': 'x\u0302j'.translate(sub_ij),
            'titlefont': {'size': 20},
            'tickfont': {'size': 18},
            'tickprefix': "   ",
            # 'showline': True,
            'linecolor': 'black',
            'zeroline': False,
            'mirror': True,
            'type': 'linear',
            # 'autorange': True,
            # 'automargin': True,
            # 'rangemode': 'tozero',
            'range': [vrg_innov.normalize_to_range[0] * wl, vrg_innov.normalize_to_range[1] * wu],
        },
        margin={'l': 50, 'b': 50, 't': 50, 'r': 50},
        height=550,
        width=600,
        legend=dict(orientation="v",
                    x=0.5, y=0.95, xanchor='left', font={'size': 20}, bordercolor="Black", borderwidth=1),
        hovermode='closest',
    )
    legendgroup += 1
    plaw_evolution_plot['layout'] = copy.copy(return_data['layout'])

    # print(return_data)

    # if plaw_evolution_var_pair is not None:
    #     plaw_evolution_plot['data'] += \
    #         [
    #             go.Scatter(
    #                 x=xi,
    #                 y=xj,
    #                 mode='lines',
    #                 name=f"x\u0302{i} * x\u0302{j}".translate(sub) + f"{np.round(b, decimals=2)}".translate(sup) +
    #                      f" = {np.round(c, decimals=2)}",
    #                 marker={
    #                     'size': 10,
    #                     'opacity': 0.5,
    #                     'line': {'width': 0.5, 'color': 'white'}
    #                 },
    #                 showlegend=True,
    #                 # legendgroup=f'group{legendgroup}'
    #             ),
    #             go.Scatter(
    #                 x=x_nd_normalized[:, i],
    #                 y=x_nd_normalized[:, j],
    #                 mode='markers',
    #                 # name=f"Actual x\u0302{i}, x\u0302{j}".translate(sub),
    #                 name=f"Offspring ND set".translate(sub),
    #                 marker={
    #                     # 'size': 10,
    #                     'opacity': 0.5,
    #                     'symbol': 'x'
    #                 },
    #                 showlegend=True,
    #                 # legendgroup=f'group{legendgroup}'
    #             ),
    #         ]

    return return_data  # , plaw_evolution_plot


@app.callback(
    Output('pcp-interactive', 'figure'),
    [Input('cross-filter-gen-slider', 'value'),
     Input(component_id='objective-space-scatter', component_property='selectedData')]
)
def update_pcp(selected_gen, selected_data):
    nearest_gen_value, x, obj, constr, rank, obj_label = get_current_gen_data(selected_gen)
    f_nd = obj[rank == 0, :]
    g_nd = constr[rank == 0, :]
    x_nd = x[rank == 0, :]
    x_dominated = x[rank > 0, :]
    f_dominated = obj[rank > 0, :]
    n_obj = obj.shape[1]
    n_constr = constr.shape[1]

    solution_id = np.array([f"{nearest_gen_value}_{indx}_r{rank[indx]}" for indx in range(obj.shape[0])])
    solution_id_nd = solution_id[rank == 0]

    data_arr = []
    data_arr_f = []
    data_arr_g = []
    # print(selected_data)
    id_indx_arr = []
    if selected_data is not None:
        for data in selected_data['points']:
            # print(data)
            data_id = data['customdata']
            if data_id[-2:] != 'r0':
                continue
            id_indx = np.where(solution_id_nd == data_id)[0][0]
            data_arr.append(x_nd[id_indx, :].tolist())
            data_arr_f.append(f_nd[id_indx, :].tolist())
            data_arr_g.append(g_nd[id_indx, :].tolist())
            id_indx_arr.append(id_indx)

        data_arr = np.array(data_arr)
        data_arr_f = np.array(data_arr_f)
        data_arr_g = np.array(data_arr_g)
    else:
        data_arr = x_nd
        data_arr_f = f_nd
        data_arr_g = g_nd
    data_list = []

    # Add objectives
    # FIXME: Plotly only shows max. 60 dimension PCP
    w = 0.05
    for i in range(n_obj):
        label = f"f{i + 1}".translate(sub)
        min_f, max_f = np.min(data_arr_f, axis=0), np.max(data_arr_f, axis=0)
        data_list.append(dict(range=[np.round(min_f[i] - w * np.abs(min_f[i]), decimals=2),
                                     np.round(max_f[i] + w * np.abs(max_f[i]), decimals=2)],
                              label=label, values=data_arr_f[:, i]),
                         )

    # Add variables
    for i in range(data_arr.shape[1]):
        data_list.append(dict(range=[xl[i], xu[i]],
                              # constraintrange=[0.6, 4.5],
                              label=f"x{i}".translate(sub), values=data_arr[:, i]
                              )
                         )

    # Add constraints
    for i in range(n_constr):
        # w_l, w_u = 0.95, 1.05
        # w_l, w_u = 1.2, 0.8
        label = f"g{i}".translate(sub)
        min_g, max_g = np.min(data_arr_g, axis=0), np.max(data_arr_g, axis=0)
        data_list.append(dict(range=[np.round(min_g[i] - w * np.abs(min_g[i]), decimals=2),
                                     np.round(max_g[i] + w * np.abs(max_g[i]), decimals=2)],
                              label=label, values=data_arr_g[:, i]
                              ),
                         )
    return_data = {'data': []}

    return_data['data'] += [
        go.Parcoords(
            line=dict(  # color=iris_df['FlowerType'],
                color='red',
                colorscale='Electric',
                showscale=True,
                # cmin=-4000,
                # cmax=-100
            ),
            dimensions=list(data_list),
            tickfont=dict(size=18),
            labelfont=dict(size=24),
            rangefont=dict(size=12),
        )]

    xaxis = {
        # 'title': obj_label[0],
        # 'title': 'Mass',
        # 'titlefont': {'size': 18},
        # 'tickfont': {'size': 16},
        # 'showline': True,
        'linecolor': 'black',
        # 'zeroline': False,
        'mirror': True,
        # 'type': 'linear',
        # 'autorange': True,
        # 'automargin': True,
        # 'rangemode': 'tozero'
    }
    yaxis = {
        # 'title': obj_label[1],
        'title': 'y',
        'titlefont': {'size': 18},
        'tickfont': {'size': 16},
        'tickprefix': "   ",
        # 'showline': True,
        'linecolor': 'black',
        'zeroline': False,
        'mirror': True,
        'type': 'linear',
        'autorange': True,
        'automargin': True,
        # 'rangemode': 'tozero'
    }

    return_data['layout'] = go.Layout(
        # title=dict(text="My Title", pad=dict(b=90, l=130, r=50)),
        # xaxis=xaxis,
        # yaxis=yaxis,
        # margin={'l': 50, 'b': 30, 't': 10, 'r': 0},
        # height=400,
        # legend=dict(orientation="v",
        #             x=0.6, y=0.95, xanchor='left', font={'size': 14}, bordercolor="Black", borderwidth=1),
        # hovermode='closest',
    )

    return return_data


@app.callback(
    Output(component_id='design-fig', component_property='figure'),
    [Input(component_id='objective-space-scatter', component_property='clickData'),
     Input('inequality-rule-checklist', 'value'),
     Input('power-law-rule-checklist', 'value')]
)
def update_design_plot(click_data, var_grp_str, power_law):
    if args.special_flag is None or click_data is None:
        return blank_fig()
    solution_id = click_data['points'][0]['customdata']
    # print("solution_id=", solution_id)
    if solution_id == "":
        return {'data': [], 'layout': None}
    print("Selected data for design plot ", solution_id)

    with h5py.File(hdf_file, 'r') as hf:
        current_gen, pop_indx, rank = solution_id.split("_")
        current_gen_data = hf[f'gen{current_gen}']
        x = np.array(current_gen_data['X'])
    x_click_selected = x[int(pop_indx), :]

    # KLUGE: VERY BIG KLUGE!!
    n_var = x.shape[1]
    if n_var == 279 or n_var == 86:
        n_shape_var = 19
    elif n_var == 579 or n_var == 176:
        n_shape_var = 39
    elif n_var == 879 or n_var == 266:
        n_shape_var = 59
    elif n_var == 129: # or n_var == 266:
        n_shape_var = 9
    else:
        return {'data': [], 'layout': None}

    from scalable_truss.truss.generate_truss import gen_truss
    from scalable_truss.truss.truss_problem_general import TrussProblemGeneral
    if n_var == 129 or n_var == 279 or n_var == 579 or n_var == 879:
        symmetry = ()
        shape_var = x_click_selected[-n_shape_var:]
        shape_var[:n_shape_var // 2 + 1] = np.sort(shape_var[:n_shape_var // 2 + 1])
        # shape_var[n_shape_var//2 + 1:] = np.flip(np.sort(shape_var[n_shape_var//2 + 1:]))
        shape_var[n_shape_var // 2 + 1:] = np.flip(
            shape_var[:n_shape_var // 2 + 1] + 0.001 + np.random.random() * 0.001)[1:]
        shape_var[n_shape_var // 2] = (shape_var[n_shape_var // 2] + shape_var[n_shape_var // 2 - 1]) / 2
        coordinates, connectivity, fixed_nodes, load_nodes, member_groups = gen_truss(n_shape_nodes=n_shape_var)
        coordinates = TrussProblemGeneral.set_coordinate_matrix(
            coordinates=coordinates, shape_var=shape_var,
            n_shape_var=n_shape_var, shape_var_mode='l', symmetry=symmetry
        )
    else:
        symmetry = ('xz', 'yz')
        shape_var = x_click_selected[-(n_shape_var // 2 + 1):]
        shape_var = np.sort(shape_var)
        coordinates, connectivity, fixed_nodes, load_nodes, member_groups = gen_truss(n_shape_nodes=n_shape_var)
        coordinates = TrussProblemGeneral.set_coordinate_matrix(
            coordinates=coordinates, shape_var=shape_var,
            n_shape_var=n_shape_var, shape_var_mode='l', symmetry=symmetry
        )

    return_data = {'data': []}

    return_data['data'] += [
        go.Scatter3d(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            z=coordinates[:, 2],
            mode='markers',
            # name='Nodes',
            marker={
                'size': 2,
                'opacity': 0.5,
                'color': 'blue',
                # 'line': {'width': 0.5, 'color': 'blue'}
            },
        )
    ]

    for i, nodes in enumerate(connectivity):
        return_data['data'] += [
            go.Scatter3d(
                x=[coordinates[int(nodes[0] - 1), 0], coordinates[int(nodes[1] - 1), 0]],
                y=[coordinates[int(nodes[0] - 1), 1], coordinates[int(nodes[1] - 1), 1]],
                z=[coordinates[int(nodes[0] - 1), 2], coordinates[int(nodes[1] - 1), 2]],
                mode='lines',
                line=dict(
                    color='black',
                    width=1),
                # name='Population',
                # marker={
                #     'size': 2,
                #     'opacity': 0.5,
                #     'color': 'blue',
                #     # 'line': {'width': 0.5, 'color': 'blue'}
                # },
            )
        ]
    var_grp = copy.deepcopy(var_grp_str)
    for i, grp in enumerate(var_grp):
        var_grp[i] = convert_checklist_str_to_list(var_grp[i])[0].split(' ')
        for j in range(len(var_grp[i])):
            var_grp[i][j] = int(var_grp[i][j])
    for indx, grp in enumerate(var_grp):
        member_indx = [grp[0], grp[1]]
        for k in member_indx:
            member_color = 'blue'
            nodes = connectivity[k, :]
            return_data['data'] += [
                go.Scatter3d(
                    x=[coordinates[int(nodes[0] - 1), 0], coordinates[int(nodes[1] - 1), 0]],
                    y=[coordinates[int(nodes[0] - 1), 1], coordinates[int(nodes[1] - 1), 1]],
                    z=[coordinates[int(nodes[0] - 1), 2], coordinates[int(nodes[1] - 1), 2]],
                    mode='lines',
                    line=dict(
                        color=member_color,
                        width=8),
                )
            ]

    for indx, law_str in enumerate(power_law):
        law = convert_checklist_str_to_list(law_str)
        member_color = 'red'
        if len(law) == 4:
            i, j, b, c = law
            i = int(i)
            j = int(j)
            b = float(b)
            c = float(c)
            member_indx = [i, j]
        elif len(law) == 2:
            i, mean_i_normalized = law
            i = int(i)
            mean_i_normalized = float(mean_i_normalized)
            member_indx = [i]
            member_color = 'blue'
        else:
            member_indx = []
        for k in member_indx:
            # FIXME: The following condition shouldnt happen. We have to make sure variable numbers are mapped
            #  properly to beams
            connectivity_indx = k
            # If shape vars are selected, highlight the corresponding members in a different color
            if k >= connectivity.shape[0]:
                connectivity_indx = member_groups['straight_xz'][1][k - connectivity.shape[0]]
                member_color = 'green'
            nodes = connectivity[connectivity_indx, :]
            return_data['data'] += [
                go.Scatter3d(
                    x=[coordinates[int(nodes[0] - 1), 0], coordinates[int(nodes[1] - 1), 0]],
                    y=[coordinates[int(nodes[0] - 1), 1], coordinates[int(nodes[1] - 1), 1]],
                    z=[coordinates[int(nodes[0] - 1), 2], coordinates[int(nodes[1] - 1), 2]],
                    mode='lines',
                    line=dict(
                        color=member_color,
                        width=8),
                )
            ]

    return_data['layout'] = go.Layout(
        # width=1200,
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=3, y=3, z=1),
        ),
        margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
        # aspectratio=dict(x=1, y=0.5, z=0.5),
        scene=go.layout.Scene(aspectratio=dict(x=3, y=1, z=1),
                              xaxis_visible=False,
                              yaxis_visible=False,
                              zaxis_visible=False,
                              xaxis_showticklabels=False,
                              yaxis_showticklabels=False,
                              zaxis_showticklabels=False),
        showlegend=False,
        xaxis={
            # 'title': 'x\u0302i'.translate(sub_ij),
            # 'titlefont': {'size': 18},
            # 'tickfont': {'size': 16},
            # # 'showline': True,
            # 'linecolor': 'black',
            # 'zeroline': False,
            # 'mirror': True,
            # 'type': 'linear',
            # 'autorange': True,
            # 'automargin': True,
            # 'rangemode': 'tozero',
            # 'automargin': True,
            'range': [0, np.max(coordinates[:, 0] + 1)],
            # 'visible': False,
            # 'showticklabels': False,
        },
        yaxis={
            # 'automargin': True,
            'range': [-5, np.max(coordinates[:, 1] + 10)],
            # 'visible': False,
            # 'showticklabels': False,
        },
        # zaxis={
        # 'automargin': True,
        #     'range': [np.min(coordinates[:, 2]) - 5, np.max(coordinates[:, 2] + 5)],
        # }
    )

    return return_data


@app.callback(
    Output(component_id='dummy_rule_rank', component_property='children'),
    Input(component_id='set_rank_power', component_property='value'),
    [State('power-law-rule-checklist', 'value'),
     State(component_id='cross-filter-gen-slider', component_property='value')]
)
def set_rule_preference(rank_power_law, power_law, current_gen):
    # print(f"Triggered, rank = {rank_power_law}")
    if current_gen != gen_arr[-1]:
        return
    for indx, law_str in enumerate(power_law):
        law = convert_checklist_str_to_list(law_str)
        # Power law
        if len(law) == 4:
            i, j, b, c = law
            i = int(i)
            j = int(j)
            # b = float(b)
            # c = float(c)
            with open(os.path.join(args.result_path, USER_INTERACT_DIR,
                                   f'{POWER_LAW_RANK_FILE_PREFIX}{current_gen}'), 'a') as fp:
                fp.write(f'{rank_power_law},{i},{j}\n')
        # Constant rule
        elif len(law) == 2:
            i, const_c = int(law[0]), float(law[1])
            with open(os.path.join(args.result_path, USER_INTERACT_DIR,
                                   f'{CONSTANT_RULE_RANK_FILE_PREFIX}{current_gen}'), 'a') as fp:
                fp.write(f'{rank_power_law},{i}\n')


if __name__ == '__main__':
    if args.port is not None:
        app.run_server(debug=True, port=int(args.port))
    else:
        app.run_server(debug=True)
