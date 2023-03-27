from dash import html, dash_table


def get_rule_table_checklists():
    return [
        {'label': 'Normalized to [1,2]', 'value': 'normalized_rule', 'disabled': False},
    ]


def get_dash_rule_table(rule_df, table_id):
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
                hidden_columns=[],
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
                css=[
                    {'selector': '.dash-spreadsheet-menu',
                     'rule': 'position:absolute;bottom:-30px'},  # move below table
                ]
            ),
        ], style={'font-size': '1.5em',
                  'overflow': 'scroll'})
