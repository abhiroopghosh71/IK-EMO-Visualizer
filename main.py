import copy
import os
import pickle
import sys
from signal import signal, SIGINT
import tempfile

import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
# from dash.exceptions import PreventUpdate
import numpy as np
import plotly.graph_objs as go
import h5py
import networkx as nx
import pandas as pd

from gui.layout import get_gen_slider_steps, blank_fig, get_hv_fig, update_hv_progress, \
    get_current_gen_data, construct_layout, POWER_LAW_TABLE_COLUMNS
from innovization.vrg_innovization import VRGInnovization
from utils.record_data import INNOVIZATION_DIR, USER_INTERACT_DIR, \
    POWER_LAW_RANK_FILE_PREFIX, CONSTANT_RULE_RANK_FILE_PREFIX
# from utils.general import get_repair_agent
from query import DemoQuery, QUERY, JSONQuery

from utils.file_io import open_file_selection_dialog
from utils.user_input import get_argparser

sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
sup = str.maketrans("0123456789.-", "⁰¹²³⁴⁵⁶⁷⁸⁹⋅⁻")
sub_ij = str.maketrans("ij", "ᵢⱼ")
# PLAY_ICON = '\u25B6'
PLAY_ICON = '\u23F5'
STOP_ICON = "\u23F9"
PAUSE_ICON = '\u23F8'
REFRESH_ICON = '\u21BA'
SAVE_ICON = '\U0001F4BE'

# Parse input arguments
args = get_argparser().parse_args()

if args.X_file is not None and args.F_file is not None and args.params_file is not None:
    query = JSONQuery(x_file=args.X_file, f_file=args.F_file,
                      param_file=args.params_file)
else:
    query = DemoQuery(args.result_path)

temp_dir = tempfile.gettempdir()
temp_path = os.path.join(temp_dir, 'optim_state_temp.hdf5')

gen_arr, latest_innov_gen_key, latest_innov_gen, xl, xu, ignore_vars = [], None, None, [], [], []

max_gen = query.get(QUERY['MAX_ITER'])
power_law_max_error = 1


def update_global_parameters():
    global gen_arr, latest_innov_gen_key, xl, xu, ignore_vars, latest_innov_gen

    latest_innov_gen = query.get(QUERY['INNOV_LATEST_GEN'])
    latest_innov_gen_key = f"gen{latest_innov_gen}"
    xl = query.get(QUERY['X_MIN'])
    xu = query.get(QUERY['X_MAX'])
    ignore_vars = query.get(QUERY['VARS_IGNORED'])
    gen_arr = query.get('GEN_ARR')


update_global_parameters()
default_pause_play_icon = PAUSE_ICON
# if os.path.exists(os.path.join(args.result_path, '.pauserun')):
#     default_pause_play_icon = PLAY_ICON
#
#
# if (latest_innov_gen is not None) and (latest_innov_gen > 0):
#     with open(os.path.join(args.result_path, INNOVIZATION_DIR,
#                            f'innov_{latest_innov_gen_key}.pkl'), 'rb') as innov_fp:
#         latest_innov = pickle.load(innov_fp)


def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Closing HDF file. Exiting.')
    # hf.close()
    # print("Closed HDF5 file.")
    if os.path.exists(temp_path):
        os.remove(temp_path)
        print("Removed temporary HDF file.")
    sys.exit(0)


signal(SIGINT, handler)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', 'static/reset.css',
                        'https://fonts.googleapis.com/icon?family=Material+Icons']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

config = {"toImageButtonOptions": {"width": None, "height": None, "format": 'svg'}}
config_heatmap = {"toImageButtonOptions": {"width": None, "height": None, "format": 'svg'}, 'editable': True}

# optim_progress_div = html.Div(id="loading-output-2")
prob_options = [{'label': 'Truss', 'value': 'Truss'}, {'label': 'OPF', 'value': 'OPF'}]

# Define HTML layout
app.layout = html.Div(construct_layout(args, gen_arr, query))


# @app.callback(Output('confirm-write', 'displayed'),
#               Input('save-data', 'n_clicks'))
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


# @app.callback(Output("hv-evolution-graph", "figure"),
#                # Output('play_optim', 'disabled')],
#               Input('optim-progress-update-interval', 'n_intervals'),
#               # State("optim-loading", "loading_state"),
#               # State("play_optim", "n_clicks"),
#               )
def update_optim_progress(n_intervals):
    """Display the number of generations completed."""
    # global prob_options
    update_global_parameters()
    hv_list = update_hv_progress(gen_arr, query)
    hv_fig = get_hv_fig(hv_list, max_gen)
    print(f"Updating HV figure after {n_intervals} intervals")

    return hv_fig


@app.callback(
    Output(component_id='generation-no', component_property='children'),
    [Input(component_id='cross-filter-gen-slider', component_property='value')]
)
def update_generation_no(selected_gen):
    return f'Generation {selected_gen}'


# @app.callback(
#     # Output(component_id='cross-filter-gen-slider', component_property='marks'),
#     Output(component_id='slider-div', component_property='children'),
#     Input(component_id='refresh-data', component_property='n_clicks'),
#     State(component_id='cross-filter-gen-slider', component_property='value')
#
# )
def refresh_dashboard(n_clicks, slider_val):
    update_global_parameters()
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


def get_innovization(current_gen, data_arr, const_tol=1e-3, rerun=False):
    # Learn power laws and constant vars from selected data.
    var_groups = [np.arange(data_arr.shape[1]).tolist()]
    innov = VRGInnovization(n_var=data_arr.shape[1], groups=var_groups, const_tol=const_tol,
                            xl=np.array(xl), xu=np.array(xu),
                            power_law_normalized=False, agent_names=['power_law_rep_sig_0'],
                            max_error=power_law_max_error)
    innov_normalized = VRGInnovization(n_var=data_arr.shape[1], groups=var_groups, const_tol=const_tol,
                                       xl=np.array(xl), xu=np.array(xu),
                                       power_law_normalized=True, agent_names=['power_law_rep_sig_0'],
                                       max_error=power_law_max_error)
    innov.learn(data_arr)
    innov_normalized.learn(data_arr)

    return innov, innov_normalized


@app.callback(
    [Output(component_id='var-group-selector', component_property='options'),
     Output(component_id='var-group-selector', component_property='value')],
    Input('cross-filter-gen-slider', 'value'),
    State("const_tol", "value")
)
def update_var_group_list(selected_gen, const_tol):
    nearest_gen_value, x, obj, constr, rank, obj_label = get_current_gen_data(selected_gen, gen_arr, query)
    x_nd = x[rank == 0, :]
    vrg_innov, vrg_innov_normalized = get_innovization(nearest_gen_value, x_nd, const_tol)
    var_grp = vrg_innov_normalized.groups
    var_group_data = []

    for i in range(len(var_grp)):
        var_group_data.append({'label': f'Group {i + 1}',
                               'value': f'{i}'})

    return var_group_data, ['0']


@app.callback(
    [Output(component_id='constant-rule-checklist', component_property='options'),
     Output(component_id='constant-rule-checklist', component_property='value')],
    [Input('cross-filter-gen-slider', 'value'),
     Input("const_tol", "value"),
     Input("minscore_constant", "value"),
     Input(component_id='objective-space-scatter', component_property='selectedData'),
     Input("constant-rule-select-all", "value"),
     Input('var-group-selector', 'value')],
    [State(component_id='constant-rule-checklist', component_property='options')]
)
def update_constant_rule_checklist(selected_gen, const_tol, minscore_constant,
                                   selected_data,
                                   constant_rule_all_selected,
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
    nearest_gen_value, x, obj, constr, rank, obj_label = get_current_gen_data(selected_gen, gen_arr, query)
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

    innov, innov_normalized = get_innovization(nearest_gen_value, data_arr, const_tol, rerun=True)
    data_arr_normalized = innov_normalized.normalize_data(data_arr)

    const_var_list = np.where(innov_normalized.relation[0].const_var_flag == 1)[0]
    const_c = innov_normalized.relation[0].c
    constant_rule_data = []

    # Show rules of type x = constant.
    print("Var grp = ", innov_normalized.groups[v_grp])
    print("const_var_list = ", const_var_list)
    for i in range(len(innov_normalized.groups[v_grp]) - 1):
        var_i = innov_normalized.groups[v_grp][i]
        if var_i in const_var_list:
            diff_i = np.abs(data_arr_normalized[:, var_i] - const_c[var_i])
            score_const = len(np.where(diff_i <= innov_normalized.relation[0].const_tol)[0]) / data_arr.shape[0]
            const_c_original = innov_normalized.xl[var_i] + (const_c[var_i] - innov_normalized.normalize_to_range[0]) / (
                        innov_normalized.normalize_to_range[1] - innov_normalized.normalize_to_range[0]) * (innov_normalized.xu[var_i] - innov_normalized.xl[var_i])
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
    [
        Output('datatable-row-ids', 'selected_rows'),
        Output('power-law-select-all', 'value')
    ],
    [
        Input('power-law-select-all', 'value'),
        Input('datatable-row-ids', 'selected_rows')
    ],
    [
        State('datatable-row-ids', 'derived_virtual_data'),
    ]
)
def select_all_power_law_rules(checked_settings, selected_rows, power_law_rows_all):
    if ctx.triggered_id == 'power-law-select-all':
        if 'select_all' in checked_settings:
            return list(range(len(power_law_rows_all))), checked_settings
        else:
            return [], checked_settings
    elif ctx.triggered_id == 'datatable-row-ids' and len(selected_rows) != len(power_law_rows_all):
        return selected_rows, []
    else:
        return selected_rows, checked_settings


# @app.callback(
#     Output('power-law-select-all', 'value'),
#     Input('datatable-row-ids', 'selected_rows'),
#     [
#         State('datatable-row-ids', 'derived_virtual_data'),
#         State('power-law-select-all', 'value')
#      ]
# )
def unselect_select_all(selected_rows, power_law_rows_all, select_all_checkbox):
    """This function will uncheck the select all checkbox if it was checked and one of the rule table rows
    was manually de-selected."""
    if 'select_all' in select_all_checkbox and len(selected_rows) == len(power_law_rows_all):
        return []


@app.callback(
    Output('datatable-row-ids', 'data'),
    [Input('cross-filter-gen-slider', 'value'),
     Input(component_id='objective-space-scatter', component_property='selectedData'),
     Input('var-group-selector', 'value'),
     Input('power-law-table-settings', 'value')],
    [
     # Power law table data
     State('datatable-row-ids', 'derived_virtual_data'),
     State('datatable-row-ids', 'derived_virtual_selected_rows')
     ]
)
def update_power_law_rule_table(selected_gen,
                                selected_data,
                                var_grp_selected,
                                checked_settings,
                                power_law_rows_all, derived_virtual_selected_rows):
    ctx = dash.callback_context

    if ctx.triggered:
        print("Power law triggers")
        id_which_triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        print(id_which_triggered)
        if id_which_triggered == 'objective-space-scatter':
            if selected_data is not None:
                print("Power law checklist triggered by selected data")

    if var_grp_selected is None:
        v_grp = 0
    else:
        v_grp = int(var_grp_selected[0])
    nearest_gen_value, x, obj, constr, rank, obj_label = get_current_gen_data(selected_gen, gen_arr, query)

    x_nd = x[rank == 0, :]

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

    innov, innov_normalized = get_innovization(nearest_gen_value, data_arr)

    b_arr, c_arr = innov.relation[1].b, innov.relation[1].c
    b_arr_normalized, c_arr_normalized = innov_normalized.relation[1].b, innov_normalized.relation[1].c
    const_var_list = np.where(innov_normalized.relation[0].const_var_flag == 1)[0]
    print("Var grp = ", innov_normalized.groups[v_grp])

    power_law_list = []
    # For every var pair in the currently selected group
    for i in range(len(innov_normalized.groups[v_grp]) - 1):
        var_i = innov_normalized.groups[v_grp][i]
        if var_i in const_var_list:
            continue
        for j in range(i + 1, len(innov_normalized.groups[v_grp])):
            var_j = innov_normalized.groups[v_grp][j]
            if var_j in const_var_list:
                continue
            # Show rules for normal power laws.
            if var_i not in const_var_list and var_j not in const_var_list:
                power_law = [var_i, var_j, b_arr_normalized[var_i, var_j], c_arr_normalized[var_i, var_j]]

                rule_compliance_id, rule_compliance_id_power, rule_compliance_id_ineq \
                    = get_rule_compliance(x_nd=x_nd, var_grp=None, curr_gen=nearest_gen_value, power_law=[power_law],
                                          power_law_max_error=0.01, const_tol=1e-3)
                rule_compliance = len(rule_compliance_id) / data_arr.shape[0]

                # Power law list: [String representation, i, j, b_ij, c_ij, correlation, rule_compliance, mse]
                if 'normalized_rule' in checked_settings:
                    power_law_str = f"x\u0302{var_i} * x\u0302{var_j}".translate(sub)\
                                    + f"{np.round(b_arr_normalized[var_i, var_j], decimals=2)}".translate(sup)\
                                    + f" = {np.round(c_arr_normalized[var_i, var_j], decimals=2)}"
                    power_law_list.append([
                        power_law_str,
                        var_i, var_j,
                        np.round(b_arr_normalized[var_i, var_j], decimals=2),
                        np.round(c_arr_normalized[var_i, var_j], decimals=2),
                        np.round(innov_normalized.correlation[var_i, var_j], decimals=2),
                        "{:.3f}".format(rule_compliance),
                        "{:.3f}".format(innov_normalized.relation[1].evaluation_metric[var_i, var_j])
                    ])
                else:
                    power_law_str = f"x{var_i} * x{var_j}".translate(sub)\
                                    + f"{np.round(b_arr[var_i, var_j], decimals=2)}".translate(sup)\
                                    + f" = {np.round(c_arr[var_i, var_j], decimals=2)}"
                    power_law_list.append([
                        power_law_str,
                        var_i, var_j,
                        np.round(b_arr[var_i, var_j], decimals=2),
                        np.round(c_arr[var_i, var_j], decimals=2),
                        np.round(innov.correlation[var_i, var_j], decimals=2),
                        "{:.3f}".format(rule_compliance),
                        "{:.3f}".format(innov.relation[1].evaluation_metric[var_i, var_j])
                    ])

    # Data frame to be used in the power law table on the display
    power_law_df = pd.DataFrame(data=power_law_list, columns=POWER_LAW_TABLE_COLUMNS)

    return power_law_df.to_dict('records')


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

    vrg_innov, vrg_innov_normalized = get_innovization(curr_gen, x_nd, const_tol)
    x_nd_normalized = vrg_innov_normalized.normalize_data(x_nd)

    n_var = x_nd.shape[1]
    rule_compliance_id_power = []
    for sol_indx, x_sol in enumerate(x_nd_normalized):
        # power_law_compliance = np.zeros([n_var, n_var])
        compliance_flag = True
        for law in power_law:
            i, j = law[:2]
            i = int(i)
            j = int(j)
            compliance = vrg_innov_normalized.relation[1].check_compliance(x_test=x_sol,
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


def parse_rule_table_selected_row(rule_table_rows_all, selected_row_indices):
    if selected_row_indices is None or len(selected_row_indices) == 0:
        return []
    # In the power law data table, for each selected row create a list [i, j, b, c] representing xi*xj^b = c
    selected_rules_from_table = np.zeros([len(selected_row_indices), 4])
    for i, row_no in enumerate(selected_row_indices):
        selected_rules_from_table[i] = list(map(rule_table_rows_all[row_no].get, ['i', 'j', 'b', 'c']))

    return selected_rules_from_table


@app.callback(
    Output('objective-space-scatter', 'figure'),
    [Input('cross-filter-gen-slider', 'value'),
     Input('inequality-rule-checklist', 'value'),
     # Data obtained on clicking a point or selecting one or more points
     Input(component_id='objective-space-scatter', component_property='selectedData'),
     Input(component_id='objective-space-scatter', component_property='clickData'),
     # Power law table data
     Input('datatable-row-ids', "derived_virtual_data"),
     Input('datatable-row-ids', "derived_virtual_selected_rows")
     ]
)
def update_objective_space_scatter_graph(selected_gen, var_grp_str, selected_data, click_data,
                                         power_law_rows_all, derived_virtual_selected_rows,
                                         power_law_max_error=0.01, const_tol=1e-3):
    ctx = dash.callback_context

    # In the power law data table, for each selected row create a list [i, j, b, c] representing xi*xj^b = c
    selected_power_law_rows = parse_rule_table_selected_row(rule_table_rows_all=power_law_rows_all,
                                                            selected_row_indices=derived_virtual_selected_rows)

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
        elif id_which_triggered == 'inequality-rule-checklist':
            # layout_update = False
            if selected_data is not None and len(var_grp) == 0:
                print("Inequality trigger obj graph")
                raise dash.exceptions.PreventUpdate

    nearest_gen_value, x, obj, constr, rank, obj_label = get_current_gen_data(selected_gen, gen_arr, query)
    f_nd = obj[rank == 0, :]
    x_nd = x[rank == 0, :]
    # x_dominated = x[rank > 0, :]
    f_dominated = obj[rank > 0, :]
    n_obj = obj.shape[1]

    solution_id = np.array([f"{nearest_gen_value}_{indx}_r{rank[indx]}" for indx in range(obj.shape[0])])
    solution_id_nd = solution_id[rank == 0]

    rule_compliance_id, rule_compliance_id_power, rule_compliance_id_ineq \
        = get_rule_compliance(x_nd=x_nd, var_grp=var_grp, curr_gen=nearest_gen_value,
                              power_law=selected_power_law_rows,
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
            nearest_gen_value, x, obj, constr, rank, obj_label = get_current_gen_data(gen, gen_arr, query)
            training_data_gen_indx = nearest_gen_value - 1  # Training data is the previous element in gen_arr

            f_nd = obj[rank == 0, :]
            x_nd = x[rank == 0, :]
            n_var = x_nd.shape[1]
            vrg_innov, vrg_innov_normalized = get_innovization(nearest_gen_value, x_nd, const_tol)

            b, c = vrg_innov_normalized.relation[1].b[i, j], vrg_innov_normalized.relation[1].c[i, j]
            ll_i, ul_i = vrg_innov_normalized.normalize_to_range[0], vrg_innov_normalized.normalize_to_range[1]
            ll_j, ul_j = vrg_innov_normalized.normalize_to_range[0], vrg_innov_normalized.normalize_to_range[1]

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
            'range': [vrg_innov_normalized.normalize_to_range[0] * wl, vrg_innov_normalized.normalize_to_range[1] * wu],
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
            'range': [vrg_innov_normalized.normalize_to_range[0] * wl, vrg_innov_normalized.normalize_to_range[1] * wu],
        },
        margin={'l': 50, 'b': 30, 't': 10, 'r': 0},
        height=400,
        legend=dict(orientation="v",
                    x=0.75, y=0.95, xanchor='left', font={'size': 14}, bordercolor="Black", borderwidth=1),
        hovermode='closest',
    )

    return return_data


# @app.callback(
#     Output(component_id='pause-continue-optimization', component_property='children'),
#     Input(component_id='pause-continue-optimization', component_property='n_clicks'),
#     State(component_id='pause-continue-optimization', component_property='children'),
# )
def toggle_pause_button(pause_click, title):
    ctx = dash.callback_context
    pause_file_path = os.path.join(args.result_path, '.pauserun')
    if ctx.triggered:
        # print(ctx.triggered)
        id_which_triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        print(id_which_triggered)
        if id_which_triggered == 'pause-continue-optimization':
            # When we want to pause the optimization
            if title == PAUSE_ICON:
                with open(pause_file_path, 'w') as _:
                    pass
                return PLAY_ICON
            # When we want to resume the optimization
            elif title == PLAY_ICON:
                if os.path.exists(pause_file_path):
                    os.remove(pause_file_path)
                return PAUSE_ICON
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
     Input('var-group-selector', 'value'),
     ],
    [State(component_id='objective-space-scatter', component_property='selectedData'),
     # Power law table data
     State('datatable-row-ids', "derived_virtual_data"),
     State('datatable-row-ids', "derived_virtual_selected_rows")]
)
def update_vrg_plot(selected_gen, include_click, exclude_click, reset_click, var_grp_selected,
                    selected_data, power_law_rows_all, derived_virtual_selected_rows,
                    power_law_max_error=0.1, const_tol=1e-3):
    """Displays the variable relation graph."""
    nearest_gen_value, x, obj, constr, rank, obj_label = get_current_gen_data(selected_gen, gen_arr, query)

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

    innov, innov_normalized = get_innovization(nearest_gen_value, data_arr, const_tol)
    if var_grp_selected is None:
        v_grp = 0
    else:
        v_grp = int(var_grp_selected[0])
    print(f"Selected var grp = {v_grp}")
    nx_graph = copy.deepcopy(innov_normalized.vrg[v_grp].graph)  # Get the networkx plot
    node_pos = nx.circular_layout(nx_graph)

    ctx = dash.callback_context

    selected_power_law_rows = parse_rule_table_selected_row(rule_table_rows_all=power_law_rows_all,
                                                            selected_row_indices=derived_virtual_selected_rows)
    selected_vars = []
    power_law_dict = {}
    for indx, power_law in enumerate(selected_power_law_rows):
        i, j, b, c = power_law
        i = int(i)
        j = int(j)
        b = float(b)
        c = float(c)
        selected_vars.append([i, j])

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
    edge_trace_list = []  # For varying the edge thickness according to rule rule_compliance
    rule_compliance_list = []
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

        power_law = [edge[0], edge[1], innov_normalized.relation[1].b[edge[0], edge[1]],
                     innov_normalized.relation[1].c[edge[0], edge[1]],
                     innov_normalized.relation[1].evaluation_metric[edge[0], edge[1]]]

        rule_compliance_id, rule_compliance_id_power, rule_compliance_id_ineq \
            = get_rule_compliance(x_nd=x_nd, var_grp=None, curr_gen=nearest_gen_value, power_law=[power_law],
                                  power_law_max_error=power_law_max_error, const_tol=const_tol)
        rule_compliance = len(rule_compliance_id) / data_arr.shape[0]
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

        edge_width = 0.5 + power_law[-1]*(5 - 0.5)
        print(edge[0], edge[1], rule_compliance, edge_width)
        edge_trace_list.append(
            go.Scatter(x=[x0, x1], y=[y0, y1], hovertemplate=f"Score = {rule_compliance}",
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
    #     innov_normalized.vrg[v_grp].graph = nx_graph'

    return fig


@app.callback(
    Output('power-law-graph', 'figure'),
    [Input('constant-rule-checklist', 'value'),
     # Power law table data
     Input('datatable-row-ids', "derived_virtual_data"),
     Input('datatable-row-ids', "derived_virtual_selected_rows")],
    [State('cross-filter-gen-slider', 'value'),
     State("const_tol", "value")]
)
def update_power_law_plot(constant_rule, power_law_rows_all, derived_virtual_selected_rows,
                          selected_gen, const_tol):
    return_data = {'data': []}
    plaw_evolution_plot = {'data': []}

    nearest_gen_value, x, obj, constr, rank, obj_label = get_current_gen_data(selected_gen, gen_arr, query)
    training_data_gen_indx = nearest_gen_value - 1  # Training data is the previous element in gen_arr

    f_nd = obj[rank == 0, :]
    x_nd = x[rank == 0, :]
    n_var = x_nd.shape[1]
    vrg_innov, vrg_innov_normalized = get_innovization(nearest_gen_value, x_nd, const_tol)
    x_nd_normalized = vrg_innov_normalized.normalize_data(x_nd)
    legendgroup = 1
    selected_power_law_rows = parse_rule_table_selected_row(rule_table_rows_all=power_law_rows_all,
                                                            selected_row_indices=derived_virtual_selected_rows)
    for indx, power_law in enumerate(selected_power_law_rows):
        i, j, b, c = power_law
        i = int(i)
        j = int(j)
        b = float(b)
        c = float(c)

        ll_i, ul_i = vrg_innov_normalized.normalize_to_range[0], vrg_innov_normalized.normalize_to_range[1]
        ll_j, ul_j = vrg_innov_normalized.normalize_to_range[0], vrg_innov_normalized.normalize_to_range[1]

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

    for indx, law_str in enumerate(constant_rule):
        law = convert_checklist_str_to_list(law_str)
        j, mean_xj = law
        j = int(j)
        mean_xj = float(mean_xj)
        ll_i, ul_i = vrg_innov_normalized.normalize_to_range[0], vrg_innov_normalized.normalize_to_range[1]
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
    xi = np.linspace(vrg_innov_normalized.normalize_to_range[0], vrg_innov_normalized.normalize_to_range[1], 100)
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
            'range': [vrg_innov_normalized.normalize_to_range[0] * wl, vrg_innov_normalized.normalize_to_range[1] * wu],
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
            'range': [vrg_innov_normalized.normalize_to_range[0] * wl, vrg_innov_normalized.normalize_to_range[1] * wu],
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
    nearest_gen_value, x, obj, constr, rank, obj_label = get_current_gen_data(selected_gen, gen_arr, query)
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


# @app.callback(
#     Output(component_id='design-fig', component_property='figure'),
#     [Input(component_id='objective-space-scatter', component_property='clickData'),
#      Input('inequality-rule-checklist', 'value'),
#      # Power law table data
#      Input('datatable-row-ids', "derived_virtual_data"),
#      Input('datatable-row-ids', "derived_virtual_selected_rows")
#      ]
# )
# def update_design_plot(click_data,
#                        var_grp_str,
#                        power_law_rows_all, derived_virtual_selected_rows):
#     if args.special_flag is None or click_data is None:
#         return blank_fig()
#     solution_id = click_data['points'][0]['customdata']
#     # print("solution_id=", solution_id)
#     if solution_id == "":
#         return {'data': [], 'layout': None}
#     print("Selected data for design plot ", solution_id)
#
#     with h5py.File(hdf_file, 'r', libver='latest', swmr=True) as hf:
#         current_gen, pop_indx, rank = solution_id.split("_")
#         current_gen_data = hf[f'gen{current_gen}']
#         x = np.array(current_gen_data['X'])
#     x_click_selected = x[int(pop_indx), :]
#
#     # KLUGE: VERY BIG KLUGE!!
#     n_var = x.shape[1]
#     if n_var == 279 or n_var == 86:
#         n_shape_var = 19
#     elif n_var == 579 or n_var == 176:
#         n_shape_var = 39
#     elif n_var == 879 or n_var == 266:
#         n_shape_var = 59
#     elif n_var == 129: # or n_var == 266:
#         n_shape_var = 9
#     else:
#         return {'data': [], 'layout': None}
#
#     from scalable_truss.truss.generate_truss import gen_truss
#     from scalable_truss.truss.truss_problem_general import TrussProblemGeneral
#     if n_var == 129 or n_var == 279 or n_var == 579 or n_var == 879:
#         symmetry = ()
#         shape_var = x_click_selected[-n_shape_var:]
#         shape_var[:n_shape_var // 2 + 1] = np.sort(shape_var[:n_shape_var // 2 + 1])
#         # shape_var[n_shape_var//2 + 1:] = np.flip(np.sort(shape_var[n_shape_var//2 + 1:]))
#         shape_var[n_shape_var // 2 + 1:] = np.flip(
#             shape_var[:n_shape_var // 2 + 1] + 0.001 + np.random.random() * 0.001)[1:]
#         shape_var[n_shape_var // 2] = (shape_var[n_shape_var // 2] + shape_var[n_shape_var // 2 - 1]) / 2
#         coordinates, connectivity, fixed_nodes, load_nodes, member_groups = gen_truss(n_shape_nodes=n_shape_var)
#         coordinates = TrussProblemGeneral.set_coordinate_matrix(
#             coordinates=coordinates, shape_var=shape_var,
#             n_shape_var=n_shape_var, shape_var_mode='l', symmetry=symmetry
#         )
#     else:
#         symmetry = ('xz', 'yz')
#         shape_var = x_click_selected[-(n_shape_var // 2 + 1):]
#         shape_var = np.sort(shape_var)
#         coordinates, connectivity, fixed_nodes, load_nodes, member_groups = gen_truss(n_shape_nodes=n_shape_var)
#         coordinates = TrussProblemGeneral.set_coordinate_matrix(
#             coordinates=coordinates, shape_var=shape_var,
#             n_shape_var=n_shape_var, shape_var_mode='l', symmetry=symmetry
#         )
#
#     return_data = {'data': []}
#
#     return_data['data'] += [
#         go.Scatter3d(
#             x=coordinates[:, 0],
#             y=coordinates[:, 1],
#             z=coordinates[:, 2],
#             mode='markers',
#             # name='Nodes',
#             marker={
#                 'size': 2,
#                 'opacity': 0.5,
#                 'color': 'blue',
#                 # 'line': {'width': 0.5, 'color': 'blue'}
#             },
#         )
#     ]
#
#     for i, nodes in enumerate(connectivity):
#         return_data['data'] += [
#             go.Scatter3d(
#                 x=[coordinates[int(nodes[0] - 1), 0], coordinates[int(nodes[1] - 1), 0]],
#                 y=[coordinates[int(nodes[0] - 1), 1], coordinates[int(nodes[1] - 1), 1]],
#                 z=[coordinates[int(nodes[0] - 1), 2], coordinates[int(nodes[1] - 1), 2]],
#                 mode='lines',
#                 line=dict(
#                     color='black',
#                     width=1),
#                 # name='Population',
#                 # marker={
#                 #     'size': 2,
#                 #     'opacity': 0.5,
#                 #     'color': 'blue',
#                 #     # 'line': {'width': 0.5, 'color': 'blue'}
#                 # },
#             )
#         ]
#     var_grp = copy.deepcopy(var_grp_str)
#     for i, grp in enumerate(var_grp):
#         var_grp[i] = convert_checklist_str_to_list(var_grp[i])[0].split(' ')
#         for j in range(len(var_grp[i])):
#             var_grp[i][j] = int(var_grp[i][j])
#     for indx, grp in enumerate(var_grp):
#         member_indx = [grp[0], grp[1]]
#         for k in member_indx:
#             member_color = 'blue'
#             nodes = connectivity[k, :]
#             return_data['data'] += [
#                 go.Scatter3d(
#                     x=[coordinates[int(nodes[0] - 1), 0], coordinates[int(nodes[1] - 1), 0]],
#                     y=[coordinates[int(nodes[0] - 1), 1], coordinates[int(nodes[1] - 1), 1]],
#                     z=[coordinates[int(nodes[0] - 1), 2], coordinates[int(nodes[1] - 1), 2]],
#                     mode='lines',
#                     line=dict(
#                         color=member_color,
#                         width=8),
#                 )
#             ]
#
#     selected_power_law_rows = parse_rule_table_selected_row(rule_table_rows_all=power_law_rows_all,
#                                                             selected_row_indices=derived_virtual_selected_rows)
#
#     for indx, power_law in enumerate(selected_power_law_rows):
#         i, j, b, c = power_law
#         i = int(i)
#         j = int(j)
#         member_indx = [i, j]
#         member_color = 'red'
#         # elif len(law) == 2:
#         #     i, mean_i_normalized = law
#         #     i = int(i)
#         #     mean_i_normalized = float(mean_i_normalized)
#         #     member_indx = [i]
#         #     member_color = 'blue'
#         # else:
#         #     member_indx = []
#         for k in member_indx:
#             # FIXME: The following condition shouldnt happen. We have to make sure variable numbers are mapped
#             #  properly to beams
#             connectivity_indx = k
#             # If shape vars are selected, highlight the corresponding members in a different color
#             if k >= connectivity.shape[0]:
#                 connectivity_indx = member_groups['straight_xz'][1][k - connectivity.shape[0]]
#                 member_color = 'green'
#             nodes = connectivity[connectivity_indx, :]
#             return_data['data'] += [
#                 go.Scatter3d(
#                     x=[coordinates[int(nodes[0] - 1), 0], coordinates[int(nodes[1] - 1), 0]],
#                     y=[coordinates[int(nodes[0] - 1), 1], coordinates[int(nodes[1] - 1), 1]],
#                     z=[coordinates[int(nodes[0] - 1), 2], coordinates[int(nodes[1] - 1), 2]],
#                     mode='lines',
#                     line=dict(
#                         color=member_color,
#                         width=8),
#                 )
#             ]
#
#     return_data['layout'] = go.Layout(
#         # width=1200,
#         scene_camera=dict(
#             up=dict(x=0, y=0, z=1),
#             center=dict(x=0, y=0, z=0),
#             eye=dict(x=3, y=3, z=1),
#         ),
#         margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
#         # aspectratio=dict(x=1, y=0.5, z=0.5),
#         scene=go.layout.Scene(aspectratio=dict(x=3, y=1, z=1),
#                               xaxis_visible=False,
#                               yaxis_visible=False,
#                               zaxis_visible=False,
#                               xaxis_showticklabels=False,
#                               yaxis_showticklabels=False,
#                               zaxis_showticklabels=False),
#         showlegend=False,
#         xaxis={
#             # 'title': 'x\u0302i'.translate(sub_ij),
#             # 'titlefont': {'size': 18},
#             # 'tickfont': {'size': 16},
#             # # 'showline': True,
#             # 'linecolor': 'black',
#             # 'zeroline': False,
#             # 'mirror': True,
#             # 'type': 'linear',
#             # 'autorange': True,
#             # 'automargin': True,
#             # 'rangemode': 'tozero',
#             # 'automargin': True,
#             'range': [0, np.max(coordinates[:, 0] + 1)],
#             # 'visible': False,
#             # 'showticklabels': False,
#         },
#         yaxis={
#             # 'automargin': True,
#             'range': [-5, np.max(coordinates[:, 1] + 10)],
#             # 'visible': False,
#             # 'showticklabels': False,
#         },
#         # zaxis={
#         # 'automargin': True,
#         #     'range': [np.min(coordinates[:, 2]) - 5, np.max(coordinates[:, 2] + 5)],
#         # }
#     )
#
#     return return_data


# @app.callback(
#     Output(component_id='dummy_rule_rank', component_property='children'),
#     Input(component_id='set_rank_power', component_property='value'),
#     [State(component_id='cross-filter-gen-slider', component_property='value'),
#      # Power law table data
#      State('datatable-row-ids', "derived_virtual_data"),
#      State('datatable-row-ids', "derived_virtual_selected_rows"),
#      ]
# )
def set_rule_preference(rank_power_law, current_gen,
                        power_law_rows_all, derived_virtual_selected_rows):
    # print(f"Triggered, rank = {rank_power_law}")
    if current_gen != gen_arr[-1]:
        return
    selected_power_law_rows = parse_rule_table_selected_row(rule_table_rows_all=power_law_rows_all,
                                                            selected_row_indices=derived_virtual_selected_rows)
    for indx, power_law in enumerate(selected_power_law_rows):
        i, j, b, c = power_law
        i = int(i)
        j = int(j)

        with open(os.path.join(args.result_path, USER_INTERACT_DIR,
                               f'{POWER_LAW_RANK_FILE_PREFIX}{current_gen}'), 'a') as fp:
            fp.write(f'{rank_power_law},{i},{j}\n')
        # Constant rule
        # elif len(law) == 2:
        #     i, const_c = int(law[0]), float(law[1])
        #     with open(os.path.join(args.result_path, USER_INTERACT_DIR,
        #                            f'{CONSTANT_RULE_RANK_FILE_PREFIX}{current_gen}'), 'a') as fp:
        #         fp.write(f'{rank_power_law},{i}\n')


if __name__ == '__main__':
    if args.port is not None:
        app.run_server(debug=True, port=int(args.port))
    else:
        app.run_server(debug=True)
