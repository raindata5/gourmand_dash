import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
from jupyter_dash import JupyterDash
import dash_core_components as dcc
from dash.dependencies import Output, Input
import pandas as pd
from pathlib import Path
import os
import plotly.graph_objects as go
import plotly.express as px
from dash.exceptions import PreventUpdate

def fig_template():
    fig = go.Figure()
    fig.layout.paper_bgcolor = '#E5ECF6'
    fig.layout.plot_bgcolor = '#E5ECF6'
    return fig

path = Path(os.getcwd())
base_dir = path.parent
data_dir = os.path.join(base_dir, "data")
data_dir

dv_runs = pd.read_csv(os.path.join(data_dir, "dv_runs.csv"), delimiter=',', quotechar='"', header='infer')
bus_cat_hold_loc_parquet = pd.read_parquet(os.path.join(data_dir, 'bus_cat_hold_loc'), engine='pyarrow')
df2_groups_pivoted_100_sorted_cut_imputed = pd.read_parquet(os.path.join(data_dir, 'df2_groups_pivoted_100_sorted_cut_imputed'), engine='pyarrow')

# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app = JupyterDash(__name__, external_stylesheets=[dbc.themes.COSMO])

app.layout = html.Div([
    html.H1('Gourmand Database',
        style={'color':'orange',
            'fontSize':'50px'}),
    html.H2('Businesses')
    ,html.Br()
    ,html.Div(id='chain_output'),
    dbc.Tabs([
        dbc.Tab([
            html.Ul([
                html.Li('Counts: '),
                html.Li('Unique: '),
                html.Li('Top Chain: '),
                html.Li('Highest Review Count: '),
            dbc.Col(
                html.Li('Computer screen <--> phone screen'), lg={'size': 3, 'offset': 
                1}, sm={'size': 7, 'offset': 
                5})
            ])
    ], label='Data at a Glance'),
        dbc.Tab([
            html.H2('Daily Review Changes', style={'textAlign': 'center'})
            ,dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Dropdown(
                                id='year-dropdown'
                                , options=[{'label': year, 'value': year} for year in bus_cat_hold_loc_parquet['CloseDate'].sort_values().unique()]
                            )
                            ,dcc.Graph(
                                id='review-delta-year-barchart'
                            )
                        ]
                    )
                    ,dbc.Col(
                        [
                            dcc.Dropdown(
                                id='state-dropdown'
                                , options=[{'label': state, 'value': state} for state in bus_cat_hold_loc_parquet['StateName'].sort_values().unique()]
                            )
                            ,dcc.Graph(
                                id='review-delta-state-barchart'
                            )
                        ]
                    )                    

                ]
            )
            ,html.H2(
                'hiya!😀'
                , id='data-heading'
                )
        ],label='Stats on States by Day')
        ,dbc.Tab([
            html.Ul([
                html.Br()
                ,html.Li([
                    'Github Repo: '
                    ,html.A('gourmand-data-pipelines', href='https://github.com/raindata5/gourmand-data-pipelines')
                ])
                ,html.Li(['Source: ',
                html.A('Yelp', href='https://www.yelp.com/developers/documentation/v3')
                ])
            ])
        ], label='Background Information')
    ,dbc.Tab([
        html.Br()
        ,dbc.Row(
            [
            dbc.Col(
                dcc.Dropdown(id = 'dag_state_dropdown_menu', options=[
                    {'label':state.title() , 'value':state} for state in list(dv_runs['dag_state'].unique())
            ])
        )
            ,dbc.Col(
                dcc.Dropdown(id = 'date_dropdown_menu',options=[
                    {'label':date , 'value':date} for date in sorted(list(dv_runs['execution_date'].unique()))
            ])
        )
            ]
        )

        ,dcc.Graph(id='runtime_chart')
        ], label = 'Airflow Task Graph'
    )
    ,dbc.Tab(
        [
            dcc.Dropdown(
                id="cond-prob-chart1-dccd"
                ,options=[{'label':state, 'value':state} for state in df2_groups_pivoted_100_sorted_cut_imputed.StateName.unique()]
            )
            ,dcc.Graph(
                id='cond-prob-chart1'
            )
            ,html.Br()
            ,dbc.Label("States")
            ,dcc.Dropdown(
                id="cond-prob-chart2-dccd"
                ,options = [{'label':state, 'value':state} for state in df2_groups_pivoted_100_sorted_cut_imputed.StateName.unique()]
                ,multi=True
                ,placeholder='Veuillez choisir un ou plusieurs pays'         
            )
            ,dcc.Graph(
                id='cond-prob-chart2'
            )
        ]
        ,label="Conditional Probabilities"
    )
    ]
    )

    
#     ,dbc.Row(
#             [
#                 dbc.Col('BusinessID',width=1 ),
#                 dbc.Col('BusinessName', width =2)
#                 ]
#             )
]
#     , style={'backgroundColor': '#E5ECF6'}
    
)

@app.callback(Output('runtime_chart','figure'),
             Input('date_dropdown_menu','value'),
             Input('dag_state_dropdown_menu','value'))
def plot_by_state_day(date, state):
    title_date = 'all days'
    title_state = 'all'
    if not date and not state:
        df_sorted = dv_runs.sort_values(['runtime_seconds'], ascending=False)
    elif state and not date :
        df = dv_runs.loc[dv_runs['dag_state'] == state]
        title_state = state
        df_sorted = df.sort_values(['runtime_seconds'], ascending=False)
    elif not state and date:
        df = dv_runs.loc[dv_runs['execution_date'] == date]
        df_sorted = df.sort_values(['runtime_seconds'], ascending=False)
        title_date = date
    else:
        df = dv_runs.loc[(dv_runs['dag_state'] == state) & (dv_runs['execution_date'] == date)]
        df_sorted = df.sort_values(['runtime_seconds'], ascending=False)
        title_state = state
        title_date = date
    fig = go.Figure()
    fig.add_bar(x=df_sorted['dag_id'], y=df_sorted['runtime_seconds'])
    fig.layout.title = f'airflow tasks runtimes for {title_date} and for {title_state} attempts'
    fig.layout.yaxis.title = 'runtime (seconds)'
    fig.layout.template = "ggplot2"
    return fig
@app.callback(
    Output('review-delta-year-barchart', 'figure')
    ,Input('year-dropdown', 'value')
)
def plot_by_year(year):
    if not year:
        raise PreventUpdate
    state_day_groups = bus_cat_hold_loc_parquet.groupby(['StateName', 'CloseDate'], as_index=False).sum()
    df=state_day_groups[state_day_groups['CloseDate'] == year].sort_values('abs_review_diff').reset_index(drop=True)
    pixel_constant= df.shape[0]
    fig = px.bar(df, y='StateName',x='abs_review_diff', title=' - '.join(['abs_review_diff', 'by State']),  orientation='h', height=200 + (20 * pixel_constant))

    return fig

@app.callback(
    Output('review-delta-state-barchart', 'figure')
    ,Input('state-dropdown', 'value')
)
def plot_by_state_day(state):
    if not state:
        raise PreventUpdate
    state_day_groups = bus_cat_hold_loc_parquet.groupby(['StateName', 'CloseDate'], as_index=False).sum()
    df=state_day_groups[state_day_groups['StateName'] == state].sort_values('abs_review_diff').reset_index(drop=True)
    pixel_constant= df.shape[0]
    fig = px.bar(df, x='CloseDate',y='abs_review_diff', title=' - '.join([f'abs_review_diff for {state}', 'each day']))

    return fig
@app.callback(
    Output('cond-prob-chart1', 'figure')
    ,Input('cond-prob-chart1-dccd', 'value')
)
def cond_prob_1(state):
    if not state:
        raise PreventUpdate
    # TODO: have a separate dataframe be placed in when no state is selected
    fig = px.bar((df2_groups_pivoted_100_sorted_cut_imputed[df2_groups_pivoted_100_sorted_cut_imputed['StateName']==state] if state else df2_groups_pivoted_100_sorted_cut_imputed),
    x=['one','two','three','four','five'],
    y='PaymentLevelName',
    hover_name='StateName',
    orientation='h',
    barmode='stack',
    height=600 ,
    title=(f'Ratings by Payment level - {state}' if state else 'Ratings by Payment level'))
    
    fig.layout.legend.orientation = 'h'
    fig.layout.legend.title = 'Rating'
    # fig.layout.legend.title = None
    fig.layout.xaxis.title = "Percent of Total Ratings"
    fig.layout.legend.x = 0.25
    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
    ))
    return fig

# make usre to plug in zeros for fig


# if __name__ == '__main__':
# app.run_server(mode='jupyterlab')

app.run_server(mode='jupyterlab')


states = ['Florida', 'Texas', 'New York', 'California']

df = df2_groups_pivoted_100_sorted_cut_imputed.loc[df2_groups_pivoted_100_sorted_cut_imputed['StateName'].isin(states)]
px.bar(
    df,
    x=['one','two','three','four','five'],
    y='PaymentLevelName',
    facet_row = 'StateName',
    hover_name='StateName',
    orientation='h',
    barmode='stack',
    height=100 + 250*len(states),
    labels={'PaymentLevelName': 'PaymentLevelName'},
    title='<br>'.join(['PaymentLevelName', ', '.join(states)])

)