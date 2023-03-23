import os

import numpy as np
from dash import html, dcc, dash_table
from plotly import graph_objs as go
import pandas as pd

# from main import gen_arr, query
from query import QUERY

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
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#5F4F93',
    'color': 'white',
    'padding': '6px'
}


def construct_layout(args, gen_arr, query):
    power_law_df = pd.DataFrame(data=[], columns=["Power law",
                                                  "i", "j", "b", "c",
                                                  "Correlation", "Score", "MSE"])
    app_mode = args.app_mode
    default_pause_play_icon = PAUSE_ICON
    if os.path.exists(os.path.join(args.result_path, '.pauserun')):
        default_pause_play_icon = PLAY_ICON

    config = {"toImageButtonOptions": {"width": None, "height": None, "format": 'svg'}}
    config_heatmap = {"toImageButtonOptions": {"width": None, "height": None, "format": 'svg'}, 'editable': True}
    html_layout = [
        html.Div([
            html.H1(children='IK-EMO Visualizer v0.1', style={'font-weight': 'normal'}),
        ],
            style={'padding': '30px 0px 20px 0px', 'background-color': '#5F4F93',  # '#059862',
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
                html.Div([html.H2(children='Scatter plot', id='scatter-heading', className='widgetTitle',
                                  style={'display': 'inline-block'})
                          ],
                         style={'padding': '0px 0px 0px 20px', 'color': '#3C4B64'}),
                html.Div([dcc.Graph(id='objective-space-scatter',
                                    hoverData={'points': [{'customdata': ''}]}, config=config)],
                         style={'padding': '0px 0px 0px 0px',  # 'background-color': 'white',
                                'margin': '0px 0px 20px 0px', 'border-bottom': '1px #EBEDEF solid',
                                'background-color': 'white'},
                         ),

                html.Div([
                    html.H6(children='Generation', id='generation-no', style={'display': 'inline-block'}),
                    html.Div([
                        html.Button(children=PLAY_ICON, id="playScatter", title='Play',
                                    style={'margin': '0 0px 0 0', 'font-size': '20px', 'border': 'none',
                                           'padding-right': '0px'}),
                        html.Button(children=STOP_ICON, id="stopScatter", title='Pause',
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
                ], style={'padding': '0px 0px 0px 0px'}, id='slider-div'),
            ], style={'width': '25%', 'display': 'inline-block', 'padding': '20px 20px 0px 20px',
                      'vertical-align': 'top', 'border': '1px solid #969696', 'border-radius': '5px',
                      'background-color': 'white',
                      'margin': '0px 0px 10px 10px'}
            ),
            html.Div([
                # Power law rules
                html.Div([
                dcc.Tabs(id="innov-rules-tab-group", value='power-law-list', children=[
                    dcc.Tab(label='Power laws', value='power-law-list', style=tab_style,
                            selected_style=tab_selected_style, children=[
                                html.Div([  # Bordered region
                                    # Rule display settings
                                    html.Div([
                                        html.Div([
                                            html.H6(children='Min. rule score', id='maxscore_power_text'),
                                            dcc.Input(id="minscore_power", type="number",
                                                      placeholder="Max. power law score",
                                                      debounce=True,
                                                      inputMode='numeric', value=0.7,
                                                      className='ruleSetting'),
                                        ]),
                                        html.Div([
                                            html.H6(children='Max. rule error', id='maxerror_power_text'),
                                            dcc.Input(id="maxerror_power", type="number",
                                                      placeholder="Max. power law error",
                                                      debounce=True,
                                                      inputMode='numeric', value=0.01,
                                                      className='ruleSetting'),
                                        ]),
                                        html.Div([
                                            html.H6(children='Min. corr.', id='mincorr_power_text'),
                                            dcc.Input(id="mincorr_power", type="number",
                                                      placeholder="Min. power corr", debounce=True,
                                                      inputMode='numeric', value=0,
                                                      className='ruleSetting'),
                                        ]),
                                        # html.Div([
                                        #     html.H5(children='Vars per rule', id='varsperrule_power_text'),
                                        #     dcc.Input(id="varsperrule_power", type="number",
                                        #     placeholder="Vars per rule",
                                        #     debounce=True,
                                        #               inputMode='numeric', value=2, disabled=True),
                                        # ]),
                                        html.Div([
                                            html.H6(children='Set rank', id='set_rank'),
                                            dcc.Input(id="set_rank_power", type="number",
                                                      debounce=True,
                                                      inputMode='numeric',
                                                      className='ruleSetting')
                                        ])
                                    ], style={'width': '30%', 'display': 'inline-block',
                                              'vertical-align': 'top'}),
                                    # Rule list
                                    html.Div([
                                        dcc.Checklist(
                                            id='power-law-select-all',
                                            options=[
                                                {'label': 'Select all', 'value': 'select_all'},
                                            ],
                                            value=[]  # ['NYC', 'MTL']
                                        ),
                                        dcc.Checklist(
                                            id='power-law-rule-checklist',
                                            options=[
                                            ],
                                            value=[]
                                        )
                                    ], style={'width': '68%', 'display': 'inline-block',
                                              'overflow': 'scroll'})
                                ], className='ruleList'),
                            ]),
                    dcc.Tab(label='Constant rules', value='constant-rule-list', style=tab_style,
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
                                            options=[
                                                # {'label': 'New York City', 'value': 'NYC'},
                                                # {'label': 'Montréal', 'value': 'MTL'},
                                                # {'label': 'San Francisco', 'value': 'SF'}
                                            ],
                                            value=[]  # ['NYC', 'MTL']
                                        )
                                    ], style={'width': '68%', 'height': '100%', 'display': 'inline-block',
                                              'overflow': 'scroll'})
                                ], className='ruleList'),
                            ]),
                    dcc.Tab(label='Inequality rules', value='inequality-rule-list', style=tab_style,
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
                    html.Div([
                        dash_table.DataTable(
                            id='datatable-row-ids',
                            columns=[
                                {'name': i, 'id': i, 'deletable': False} for i in power_law_df.columns
                                # omit the id column
                                if i != 'id'
                            ],
                            data=power_law_df.to_dict('records'),
                            editable=True,
                            filter_action="native",
                            sort_action="native",
                            sort_mode='multi',
                            row_selectable='multi',
                            row_deletable=False,
                            selected_rows=[],
                            page_action='native',
                            page_current=0,
                            page_size=10,
                            hidden_columns=["i", "j", "b", "c"],
                            style_as_list_view=False,
                            style_cell={'padding': '5px'},
                            style_header={
                                # 'backgroundColor': 'white',
                                'fontWeight': 'bold'
                            },
                            style_cell_conditional=[
                                {
                                    'if': {'column_id': c},
                                    'textAlign': 'left'
                                } for c in ['Power law']
                            ],
                        ),
                    ], style={'width': '25%', 'font-size': '1.75em'}),
                ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}),
                html.Div([
                    html.Span([html.H3(children="Variable relation graph", id='vrg-fig-heading',
                                       style={'width': '50%', 'display': 'inline-block',
                                              'padding': '10px'}, className='widgetTitle'),
                               dcc.Dropdown([], id='var-group-selector', searchable=False,
                                            style={'width': '30%', 'display': 'inline-block'}),
                               # dcc.Input(
                               #     id="vrg_vars",
                               #     type="text", placeholder="Var pairs", debounce=True,
                               #     inputMode='numeric', value=None,
                               #     style={'width': '10%', 'display': 'inline-block'}
                               # ),
                               html.Button('\u2705', id='vrg-include', n_clicks=0, title='Add edge',
                                           style={'padding': '0', 'border': 'none', 'background': 'none',
                                                  'margin-left': '10px', 'font-size': '20px'}),
                               html.Button('\u274C', id='vrg-exclude', n_clicks=0, title='Remove edge',
                                           style={'padding': '0', 'border': 'none', 'background': 'none',
                                                  'margin-left': '10px', 'font-size': '20px'}),
                               html.Button('\u21BA', id='vrg-reset', n_clicks=0, title='Reset VRG',
                                           style={'padding': '0', 'border': 'none', 'background': 'none',
                                                  'margin-left': '10px', 'font-size': '35px'}),
                               ]),
                    dcc.Graph(id='vrg-fig',
                              hoverData={'points': [{'customdata': ''}]}, config=config),
                        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top',
                                  'padding': '0 0 0 10px'}),

            ], style={'width': '69%', 'padding': '20px 20px 20px 20px', 'display': 'inline-block',
                      'vertical-align': 'top',
                      'overflow': 'scroll',
                      'margin': '0px 10px 10px 10px',
                      'border': '1px solid #969696', 'border-radius': '5px', 'background-color': 'white'}),

            html.Div([
                html.Div([

                    html.Div([html.H2(children='Power law graph', id='power-law-graph-heading',
                                      className='widgetTitle')],
                             style={'padding': '20px 20px 0px 20px', 'color': '#3C4B64', }),
                    html.Div([
                        html.Div([dcc.Graph(id='power-law-graph',
                                            hoverData={'points': [{'customdata': ''}]}, config=config)]),
                    ],
                        style={'border': '1px solid #969696', 'border-radius': '5px', 'background-color': 'white',
                               'margin': '0px 20px 20px 20px'}),
                    html.Div([
                        html.Div([
                            html.H2(children='Power law evolution', id='power-law-evolution-heading',
                                    className='widgetTitle')], style={'width': '80%', 'display': 'inline-block'}
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
            ], style={'height': '500px'}
            ),
            html.Div(id='dummy_rule_rank'),
            # Padding: top right bottom left
            html.Div([
                html.Div([html.H2(children='Parallel coordinate plot (PCP)',
                                  id='pcp-heading', className='widgetTitle')],
                         style={'padding': '0px 20px 20px 20px', 'color': '#3C4B64'}),
                html.Div([
                    html.Div([dcc.Graph(id='pcp-interactive',
                                        hoverData={'points': [{'customdata': ''}]}, config=config)],
                             style={'width': '1000%'}
                             )
                ], style={'overflow': 'scroll', 'border': '1px solid #969696',
                          'border-radius': '5px',
                          'background-color': 'white', 'padding': '0px 20px 20px 10px',
                          'margin': '0px 20px 20px 20px'}),
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
    ]

    return html_layout


def get_gen_slider_steps(gen_list):
    gen_slider_steps = {str(int(gen)): '' for gen in gen_list}

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
    # all_gen_val = gen_arr
    # Original begin
    # nearest_gen_value = int(all_gen_val[np.abs(all_gen_val - selected_gen).argmin()])
    # obj_label = query.get(QUERY['OBJ_LABELS'])
    # obj, x, rank, constr = query.get_iter_data(nearest_gen_value, 'F', 'X', 'rank', 'G')
    # Original end
    nearest_gen_value = gen_arr[0]
    x = query.get('X')
    obj = query.get('F')
    # constr = query.get('G')
    constr = np.zeros_like(x)
    # rank = query.get('rank')
    rank = np.zeros(x.shape[0])
    obj_label = query.get('obj_label')

    return nearest_gen_value, x, obj, constr, rank, obj_label
