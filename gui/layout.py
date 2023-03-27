import numpy as np
import pandas as pd
from dash import html, dcc
from plotly import graph_objs as go

from gui.dcc_elements import get_dash_rule_table, get_rule_table_checklists
from utils.postprocess.statistical_performance import calc_hv


PLAY_ICON = '\u23F5'
STOP_ICON = "\u23F9"
PAUSE_ICON = '\u23F8'
REFRESH_ICON = '\u21BA'
SAVE_ICON = '\U0001F4BE'

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#5F4F93',
    'color': 'white',
    'padding': '6px',
}
tab_disabled_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
}

POWER_LAW_TABLE_COLUMNS = ["Power law", "i", "j", "b", "c", "Corr.", "Compliance", "Metric"]


def construct_layout(args, gen_arr, query):
    power_law_df = pd.DataFrame(data=[], columns=POWER_LAW_TABLE_COLUMNS)
    app_mode = args.app_mode
    # default_pause_play_icon = PAUSE_ICON
    # if os.path.exists(os.path.join(args.result_path, '.pauserun')):
    #     default_pause_play_icon = PLAY_ICON

    config = {"toImageButtonOptions": {"width": None, "height": None, "format": 'svg'}}
    config_heatmap = {"toImageButtonOptions": {"width": None, "height": None, "format": 'svg'}, 'editable': True}
    html_layout = [
        html.Div([
            html.Div([
                html.H1(children='IK-EMO Visualizer',
                        style={'font-weight': 'normal'})],
                        style={'display': 'inline-block'}),
            html.Div([
                html.H5(children='v0.2.0.0',
                        style={'font-weight': 'normal'})
                ], style={'display': 'inline-block', 'vertical-align': 'bottom'}),
        ],
            style={'padding': '10px 0px 10px 0px', 'background-color': '#5F4F93',  # '#059862',
                   'margin': '0px 0px 20px 0px',
                   'border-bottom': '1px #EBEDEF solid',
                   # 'font-family': "Arial",
                   'color': 'white', 'text-align': 'center',
                   # 'position': 'sticky', "top": '0'
                   },
        ),
        ]

    optim_control_title = "IK-EMO Controls (disabled)"
    optim_control_disabled = True
    pause_play_hover_text = "Available in a future release"
    refresh_hover_text = "Available in a future release"
    save_hover_text = "Available in a future release"
    if app_mode == 'interactive':
        optim_control_disabled = False
        optim_control_title = "IK-EMO Controls"
        pause_play_hover_text = "Pause/continue run"
        refresh_hover_text = "Update data"
        save_hover_text = "Save changes"

    # html_layout += [
    #     # Optimization controls
    #     html.Div([
    #         html.Div([html.H2(children=optim_control_title, className='widgetTitle')],
    #                  style={'color': '#3C4B64', 'width': '25%', 'font-weight': 'normal', 'display': 'inline-block',
    #                         'vertical-align': 'middle'}),
    #
    #         html.Div([
    #             html.Div([
    #                 html.Button(default_pause_play_icon, id='pause-continue-optimization', className="button3",
    #                             disabled=optim_control_disabled,
    #                             title=pause_play_hover_text,
    #                             style={'width': '10%', 'margin': '0px 20px 0px 0px', 'display': 'inline-block',
    #                                    'font-size': '30px', 'border': 'none'}),
    #                 html.Button(REFRESH_ICON, id='refresh-data', className='button', title=refresh_hover_text,
    #                             disabled=optim_control_disabled,
    #                             style={'width': '10%', 'margin': '0px 20px 0px 0px', 'display': 'inline-block',
    #                                    'font-size': '40px', 'border': 'none'}),
    #                 html.Button(SAVE_ICON, id='save-data', className='button', title=save_hover_text,
    #                             disabled=optim_control_disabled,
    #                             style={'width': '10%', 'margin': '0px 20px 0px 0px', 'display': 'inline-block',
    #                                    'font-size': '25px', 'border': 'none'}),
    #                 dcc.ConfirmDialog(id='confirm-write',
    #                                   message='This will write the results to disk. Continue?'),
    #             ], style={'display': 'inline-block', 'width': '75%', 'vertical-align': 'middle',
    #                       'padding': '0px 0 0 0'}),
    #
    #             # html.Div([
    #                 # dcc.Loading(
    #                 #     id="optim-loading",
    #                 #     children=[html.Div([optim_progress_div])],
    #                 #     type="circle",
    #                 # ),
    #                 # html.Div([
    #                 # ], id='optim-rogress', style={'text-align': 'center'}),
    #                 # dcc.Interval(
    #                 #     id='interval-component',
    #                 #     interval=3 * 1000,  # in milliseconds
    #                 #     n_intervals=0
    #                 # )
    #             # ], style={'display': 'inline-block', 'width': '15%', 'vertical-align': 'middle'}),
    #
    #         ], style={'width': '70%', 'display': 'inline-block'}
    #         )
    #     ],
    #         style={'width': '95.5%', 'display': 'inline-block', 'vertical-align': 'middle',
    #                'padding': '0px 0px 0px 40px',
    #                'border': '1px solid #969696', 'border-radius': '5px',
    #                'margin': '0px 0px 20px 20px',
    #                'background-color': 'white'}
    #     ),
    # ]

    html_layout += [
        html.Div([
            html.Div([
                dcc.Tabs(id="var-obj-tab-group", value='scatter-plot', children=[
                    dcc.Tab(label='Scatter Plot', value='scatter-plot', style=tab_style,
                            selected_style=tab_selected_style, children=[
                                dcc.Graph(id='objective-space-scatter',
                                          hoverData={'points': [{'customdata': ''}]}, config=config),
                                html.Div([
                                    html.H6(children='Generation', id='generation-no', style={'display': 'inline-block'}),
                                    html.Div([
                                        html.Button(children=PLAY_ICON, id="playScatter", title='Play',
                                                    disabled=False,
                                                    style={'margin': '0 0px 0 0', 'font-size': '20px',
                                                           'border': 'none',
                                                           'padding-right': '0px'}),
                                        html.Button(children=STOP_ICON, id="stopScatter", title='Pause',
                                                    disabled=False,
                                                    style={'margin': '0 0px 0 0', 'font-size': '20px',
                                                           'border': 'none',
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
                                        tooltip={"placement": "bottom", "always_visible": False},
                                        marks=get_gen_slider_steps(gen_arr),
                                        # marks={str(int(gen)): str(int(gen)) for gen in gen_arr}
                                    )
                                ], style={'padding': '0px 0px 0px 0px'}, id='slider-div')
                            ]),
                    dcc.Tab(label='PCP', value='pcp-plot', style=tab_style,
                            selected_style=tab_selected_style, children=[
                                html.Div([
                                    html.Div([
                                        dcc.Graph(id='pcp-interactive',
                                                  hoverData={'points': [{'customdata': ''}]}, config=config,
                                                  # style={'width': '200px'}
                                                  )
                                            ], style={'min-width': '200%', 'width': '200%'})
                                ], style={'overflow': 'scroll'})
                            ]),
                    dcc.Tab(label='Selected design', value='selected-design', style=tab_style,
                            disabled=True, disabled_style=tab_disabled_style,
                            selected_style=tab_selected_style, children=[
                                html.Div([
                                    dcc.Graph(id='design-fig', figure=blank_fig(),
                                              hoverData={'points': [{'customdata': ''}]}, config=config,
                                              style={'width': '100%'})
                                ], style={'width': '100%', 'padding': '0px 20px 20px 20px',
                                          'border': '1px solid #969696', 'border-radius': '5px',
                                          'margin': '20px 20px 20px 20px', 'background-color': 'white'})
                                ])
                ], style={'display': 'inline-block',
                          'vertical-align': 'top',
                          'padding': '0 0 0 10px'}),


            ], style={'display': 'inline-block', 'padding': '10px 10px 0px 10px',
                      'vertical-align': 'top', 'border': '1px solid #969696', 'border-radius': '5px',
                      'background-color': 'white',
                      'margin': '0px 0px 10px 10px'},
                className='col-left'),
            html.Div([
                # Power law rules
                html.Div([
                    dcc.Tabs(id="innov-rules-tab-group", value='power-law-list', children=[
                        dcc.Tab(label='Power laws', value='power-law-list', style=tab_style,
                                disabled=False, disabled_style=tab_disabled_style,
                                selected_style=tab_selected_style, children=[
                                    html.Div([
                                        dcc.Checklist(
                                            id='power-law-select-all',
                                            options=[{'label': 'Select all', 'value': 'select_all', 'disabled': False}],
                                            value=[],
                                            inline=True
                                        )], style={'display': 'inline-block'}),
                                    html.Div([
                                        dcc.Checklist(
                                            id='power-law-table-settings',
                                            options=get_rule_table_checklists(),
                                            value=['normalized_rule'],
                                            inline=True
                                        )], style={'display': 'inline-block'}),
                                    get_dash_rule_table(rule_df=power_law_df,
                                                        table_id='datatable-row-ids')], className='ruleList'),
                        dcc.Tab(label='Constant rules', value='constant-rule-list', style=tab_style,
                                disabled=True, disabled_style=tab_disabled_style,
                                selected_style=tab_selected_style, children=[
                                    html.Div([  # Bordered region
                                        # Rule display settings
                                        html.Div([
                                            html.Div([
                                                html.H6(children='Min. rule score', id='maxscore_constant_text'),
                                                dcc.Input(id="minscore_constant", type="number",
                                                          placeholder="Max. constant rule score", debounce=True,
                                                          inputMode='numeric', value=0, className='ruleSetting'),
                                                html.Div([
                                                    html.H6(children='Const. tol.', id='const_tol_text'),
                                                    dcc.Input(id="const_tol", type="number",
                                                              placeholder="Constant rule tol.", debounce=True,
                                                              inputMode='numeric', value=0.01,
                                                              className='ruleSetting'),
                                                ]),
                                            ]),
                                        ], style={'width': '30%', 'height': '100%', 'display': 'inline-block',
                                                  'vertical-align': 'top'}),
                                        # Rule list
                                        html.Div([
                                            dcc.Checklist(
                                                id='constant-rule-select-all',
                                                options=[
                                                    {'label': 'Select all', 'value': 'select_all'},
                                                ],
                                                value=[]  # ['NYC', 'MTL']
                                            ),
                                            dcc.Checklist(
                                                id='constant-rule-checklist',
                                                options=[],
                                                value=[]
                                            )
                                        ], style={'width': '68%', 'height': '100%', 'display': 'inline-block',
                                                  'overflow': 'scroll'})
                                    ], className='ruleList'),
                                ]),
                        dcc.Tab(label='Inequality rules', value='inequality-rule-list', style=tab_style,
                                disabled=True, disabled_style=tab_disabled_style,
                                selected_style=tab_selected_style, children=[
                                    html.Div([  # Bordered region
                                        # Rule display settings
                                        html.Div([
                                            html.Div([
                                                html.H6(children='Min. rule score', id='minscore_ineq_text'),
                                                dcc.Input(id="minscore_ineq", type="number",
                                                          placeholder="Min. ineq. score",
                                                          debounce=True,
                                                          inputMode='numeric', value=0, className='ruleSetting'),
                                            ], style={'display': 'inline-block'}),
                                            html.Div([
                                                html.H6(children='Min. var. corr.', id='mincorr_ineq_text'),
                                                dcc.Input(id="mincorr_ineq", type="number",
                                                          placeholder="Min. ineq. corr",
                                                          debounce=True,
                                                          inputMode='numeric', value=0, className='ruleSetting'),
                                            ]),
                                        ], style={'width': '30%', 'height': '100%', 'display': 'inline-block',
                                                  'vertical-align': 'top'}),
                                        # Rule list
                                        html.Div([
                                            dcc.Checklist(
                                                id='ineq-select-all',
                                                options=[
                                                    {'label': 'Select all', 'value': 'select_all'},
                                                ],
                                                value=[]
                                            ),
                                            dcc.Checklist(
                                                id='inequality-rule-checklist',
                                                options=[
                                                ],
                                                value=[]
                                            )
                                        ], style={'width': '68%', 'height': '100%', 'display': 'inline-block',
                                                  'overflow': 'scroll'})
                                    ], className='ruleList')
                                ]),
                    ]),
                ], style={'width': '58%', 'display': 'inline-block', 'vertical-align': 'top'}),
                html.Div([
                    dcc.Tabs(id="innov-figures-tab-group", value='vrg', children=[
                        dcc.Tab(label='Variable Relation Graph', value='vrg', style=tab_style,
                                disabled=False, disabled_style=tab_disabled_style,
                                selected_style=tab_selected_style, children=[
                                   dcc.Dropdown([], id='var-group-selector', searchable=False,
                                                style={'width': '50%', 'display': 'inline-block',
                                                       'vertical-align': 'middle'}),
                                   # dcc.Input(
                                   #     id="vrg_vars",
                                   #     type="text", placeholder="Var pairs", debounce=True,
                                   #     inputMode='numeric', value=None,
                                   #     style={'width': '10%', 'display': 'inline-block'}
                                   # ),
                                   html.Button('\u2705', id='vrg-include', n_clicks=0, title='Add edge',
                                               style={'padding': '0', 'border': 'none', 'background': 'none',
                                                      'margin-left': '10px', 'font-size': '20px',
                                                      'display': 'inline-block',
                                                       'vertical-align': 'middle'}),
                                   html.Button('\u274C', id='vrg-exclude', n_clicks=0, title='Remove edge',
                                               style={'padding': '0', 'border': 'none', 'background': 'none',
                                                      'margin-left': '10px', 'font-size': '20px',
                                                      'display': 'inline-block',
                                                       'vertical-align': 'middle'}),
                                   html.Button('\u21BA', id='vrg-reset', n_clicks=0, title='Reset VRG',
                                               style={'padding': '0', 'border': 'none', 'background': 'none',
                                                      'margin-left': '10px', 'font-size': '35px',
                                                      'display': 'inline-block',
                                                       'vertical-align': 'middle'}),
                                   dcc.Graph(id='vrg-fig', hoverData={'points': [{'customdata': ''}]},
                                             config=config)], className='ruleList'),

                        dcc.Tab(label='Rule Plot', value='rule-plot', style=tab_style,
                                disabled=False, disabled_style=tab_disabled_style,
                                selected_style=tab_selected_style, children=[
                                    html.Div([
                                        html.Div([dcc.Graph(id='power-law-graph',
                                                            hoverData={'points': [{'customdata': ''}]},
                                                            config=config)],
                                                 style={'overflow': 'scroll'}),
                                    ],
                                        style={'border': '1px solid #969696', 'border-radius': '5px',
                                               'background-color': 'white',
                                               'margin': '0px 20px 20px 20px'})], className='ruleList'),
                        dcc.Tab(label='Rule Evolution', value='rule-evolution-plot', style=tab_style,
                                disabled=True, disabled_style=tab_disabled_style,
                                selected_style=tab_selected_style, children=[
                                    html.Div([
                                        dcc.Input(
                                            id="plaw_evolution_vars",
                                            type="text", placeholder="Var pairs", debounce=True,
                                            inputMode='numeric', value=f"0,1",
                                            style={'width': '100px'}
                                        ),
                                        html.Div([dcc.Graph(id='power-law-evolution-graph',
                                                            hoverData={'points': [{'customdata': ''}]},
                                                            config=config)])
                                    ], style={})], className='ruleList'),
                                   ]),
                        ], style={'width': '41%', 'display': 'inline-block', 'vertical-align': 'top',
                                  'padding': '0 0 0 10px', 'overflow': 'scroll'}),


            ], style={'display': 'inline-block',
                      'vertical-align': 'top',
                      'overflow': 'scroll',
                      'margin': '0px 10px 10px 10px',
                      'border': '1px solid #969696', 'border-radius': '5px', 'background-color': 'white'},
                className='col')
            ], className='row')

            # Historical data
            # html.Div([
            #     html.Div([
            #         html.Div([
            #             html.H2(children='Optimization progress', id='hv-evolution-heading',
            #                     style={'display': 'inline-block'}, className='widgetTitle'),
            #             dcc.Interval(id='optim-progress-update-interval',
            #                          interval=20 * 1000,  # in milliseconds
            #                          n_intervals=0),
            #             html.Div([
            #                 html.Div([], className='circle--outer'),
            #                 html.Div([], className='circle--inner'),
            #             ], className="video__icon", style={'display': 'inline-block'}),
            #         ]),
            #         html.Div([
            #             dcc.Graph(id='hv-evolution-graph', figure=get_hv_fig(update_hv_progress(gen_arr, query),
            #                                                                  query.get(QUERY['MAX_ITER'])),
            #                       hoverData={'points': [{'customdata': ''}]}, config=config,
            #                       style={'width': '100%'})
            #         ], style={'width': '100%', 'padding-top': '20px'})
            #     ], style={'width': '100%', 'display': 'inline-block', 'padding': '0 0 0 20px'})
            # ], style={'width': '29%', 'display': 'inline-block', 'float': 'center',
            #           'padding': '20px 20px 20px 20px',
            #           'vertical-align': 'top', 'background-color': 'white',
            #           'border': '1px solid #969696', 'border-radius': '5px', 'margin': '0px 0px 20px 20px',
            #           'overflow': 'scroll', 'height': '83%'}),
        ]

    return html_layout


def get_gen_slider_steps(gen_list):
    if len(gen_list) > 0:
        gen_slider_steps = {str(int(gen)): '' for gen in gen_list}
    else:
        gen_slider_steps = {'N/A'}

    return gen_slider_steps


def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

    return fig


def get_hv_fig(hv_list, max_gen=None):
    fig = go.Figure(go.Scatter(x=hv_list[:, 0], y=hv_list[:, 1],
                               mode='markers+lines',
                               marker={'line': {'color': 'white'}}))
    if max_gen is not None:
        fig.add_vline(x=max_gen, line_dash="dot",
                      annotation_text="Termination",
                      line_color='red',
                      annotation_position="right",
                      annotation_textangle=-90)
    else:
        print("Max gen not defined so no line drawn in HV figure.")
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


def update_hv_progress(gen_list, query):
    """Create a HV figure to denote optimization progress."""
    # pf_list = []

    # Calculate nadir point from final gen pf
    nearest_gen_value, x, obj, constr, rank, obj_label = get_current_gen_data(max(gen_list), gen_list, query)
    # final_pf = find_pf(obj[rank == 0])
    final_pf = obj[rank == 0]
    ideal_point = np.min(final_pf, axis=0)
    nadir_point = np.max(final_pf, axis=0)

    min_val = ideal_point
    max_val = nadir_point

    hv_list = []
    for gen in gen_list:
        nearest_gen_value, x, obj, constr, rank, obj_label = get_current_gen_data(gen, gen_list, query)
        pf_gen = obj[rank == 0]
        pf_normalized = (pf_gen - min_val) / (max_val - min_val)
        hv_gen = calc_hv(pf_normalized, ref_point=np.array([1.1, 1.1]))
        hv_list.append([gen, hv_gen])

    hv_list = np.array(hv_list)

    return hv_list


def get_current_gen_data(selected_gen, gen_arr, query):
    nearest_gen_value = gen_arr[0]
    x = query.get('X')
    obj = query.get('F')
    # constr = query.get('G')
    constr = np.zeros_like(x)
    # rank = query.get('rank')
    rank = np.zeros(x.shape[0])
    obj_label = query.get('obj_label')

    return nearest_gen_value, x, obj, constr, rank, obj_label
