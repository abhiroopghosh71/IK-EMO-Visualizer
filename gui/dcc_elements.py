from dash import html, dash_table, dcc
from include.constants import *


def get_rule_table_checklists():
    return [
        {'label': 'Use normalized vars x\u0302 \u2208 [1,2]', 'value': NORMALIZED_RULE, 'disabled': False},
    ]


def get_dash_rule_table(rule_df, table_id, hidden_columns=()):
    return html.Div([
            dash_table.DataTable(
                id=table_id,
                columns=[
                    {'name': i, 'id': i, 'deletable': False} for i in rule_df.columns
                    # omit the id column
                    if i != 'id'
                ],
                data=rule_df.to_dict('records'),
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
                hidden_columns=hidden_columns,
                style_as_list_view=False,
                style_cell={'padding': '5px'},
                style_header={'fontWeight': 'bold'},
                style_cell_conditional=[
                    {
                        'if': {'column_id': c},
                        'textAlign': 'left'
                    } for c in ['Power law']
                ],
                css=[
                    {'selector': '.dash-spreadsheet-menu',
                     'rule': 'position:absolute;bottom:-30px'},  # move below table
                ]
            ),
        ], style={'font-size': '1.5em',
                  'overflow': 'scroll'})


def get_rule_table_div(html_id_prefix, rule_df, normalize_enabled=False):
    """This function returns an HTML Div consisting of the rule tables and associated settings."""
    if html_id_prefix != '':
        html_id_prefix += '-'

    if normalize_enabled:
        normalize_checkbox = [NORMALIZED_RULE]
    else:
        normalize_checkbox = []

    return [
        html.Div([
            html.H5(children="Rule Search Preferences"),
            html.Div([
                dcc.Dropdown(
                    options=[
                        {'label': 'Correlation', 'value': CORRELATION},
                        {'label': 'Abs(Correlation)', 'value': ABS_CORRELATION}
                    ],
                    value=ABS_CORRELATION,
                    id=f'{html_id_prefix}rule-select-criteria'
                )
            ], style={'display': 'inline-block', 'width': '25%', 'margin': '0px 10px 0px 0px',
                      'vertical-align': 'middle'}),
            html.Div([
                dcc.Dropdown(
                    options=[
                        {'label': 'Less Than/Equals (<=)', 'value': LESS_THAN},
                        {'label': 'Greater Than/Equals (>=)', 'value': GREATER_THAN}
                    ],
                    value=GREATER_THAN,
                    id=f'{html_id_prefix}rule-select-criteria-relation'
                )
            ], style={'display': 'inline-block', 'width': '30%', 'margin': '0px 10px 0px 0px',
                      'vertical-align': 'middle'}),
            html.Div([
                dcc.Input(type='number', debounce=True, style={'width': '100%'},
                          value=DEFAULT_CORRELATION_LIMIT,
                          id=f'{html_id_prefix}rule-select-criteria-value')
            ], style={'display': 'inline-block', 'width': '10%', 'margin': '0px 10px 0px 0px',
                      'vertical-align': 'middle'}),
            html.Div([
                html.Button(id=f'{html_id_prefix}scan-rules', children="Scan Rules", className='button')
            ], style={'display': 'inline-block', 'width': '20%', 'margin': '0px 0px 0px 0px'}),
        ], style={'margin': '10px 10px 10px 10px'}),
        html.Hr(style={'margin': '20px 10px 10px 10px'}),
        html.Div([
            dcc.Checklist(
                id=f'{html_id_prefix}select-all',
                options=[{'label': 'Select all', 'value': 'select_all', 'disabled': False}],
                value=[],
                inline=True
            )], style={'display': 'inline-block'}),
        html.Div([
            dcc.Checklist(
                id=f'{html_id_prefix}table-settings',
                options=get_rule_table_checklists(),
                value=normalize_checkbox,
                inline=True,
                persistence=False
            )], style={'display': 'inline-block'}),
        get_dash_rule_table(rule_df=rule_df,
                            table_id=f'{html_id_prefix}datatable-row-ids')]
