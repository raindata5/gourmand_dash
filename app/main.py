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
import datetime
from typing import Union
import numpy as np
from random import choices

def fig_template():
    fig = go.Figure()
    fig.layout.paper_bgcolor = '#E5ECF6'
    fig.layout.plot_bgcolor = '#E5ECF6'
    return fig

def to_unix_or_reg(the_date: Union[str, int], the_format: str, option: str):
    unix_format = '%s'
    if option == 'unix':
        new_date = datetime.datetime.strptime(the_date, the_format).strftime(unix_format)
    elif option == 'reg':
        new_date = datetime.datetime.strptime(the_date, unix_format).strftime(the_format)
    elif option =='unix_reg':
        new_date = datetime.datetime.fromtimestamp(the_date).strftime(the_format)
    else:
        print('no good option')
        exit(0)
    return new_date


path = Path(os.getcwd())
base_dir = path.parent
data_dir = os.path.join(base_dir, "data")
data_dir

dv_runs = pd.read_csv(os.path.join(data_dir, "dv_runs.csv"), delimiter=',', quotechar='"', header='infer')
bus_cat_hold_loc_parquet = pd.read_parquet(os.path.join(data_dir, 'bus_cat_hold_loc'), engine='pyarrow')
df2_groups_pivoted_100_sorted_cut_imputed = pd.read_parquet(os.path.join(data_dir, 'df2_groups_pivoted_100_sorted_cut_imputed'), engine='pyarrow')
electric1 = px.colors.sequential.Electric[0]
days = sorted(bus_cat_hold_loc_parquet.CloseDate.unique().tolist())
the_format = "%Y-%m-%d"
unix_format = '%s'

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
                dcc.Dropdown(
                    id = 'dag_state_dropdown_menu', options=[
                    {'label':state.title() , 'value':state} for state in list(dv_runs['dag_state'].unique())
            ]) , md={'size': 12}, lg={'size': 5, 'offset':1}
        )
            ,dbc.Col(
                dcc.Dropdown(
                    id = 'date_dropdown_menu',options=[
                    {'label':date , 'value':date} for date in sorted(list(dv_runs['execution_date'].unique()))
            ]) , md={'size': 12}, lg={'size': 5, 'offset':0} 
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
            ,dcc.Slider(
        id='payment-lvl-slider'
                , min=0, max=4, step=1, dots=True, included=False, value=4, marks={
            0: {'label': 'Unknown', 'style': {'color': electric1, 'fontWeight': 'bold'}},
            1: {'label': 'Very Low', 'style': {'color': electric1, 'fontWeight': 'bold'}},
            2: {'label': 'Low', 'style': {'color': electric1, 'fontWeight': 'bold'}},
            3: {'label': 'High', 'style': {'color': electric1, 'fontWeight': 'bold'}},
            4: {'label': 'Very High', 'style': {'color': electric1, 'fontWeight': 'bold'}}
                }
            )
            ,html.Br()
            ,dcc.Slider(
                id='close-data-slider'
                ,min=int(to_unix_or_reg(days[::2][0], the_format, 'unix')) - 86400
                ,max=int(to_unix_or_reg(days[::2][-1], the_format, 'unix')) + 86400
                ,step=86400
                ,dots=True
                ,included=False
#                 ,value= int(to_unix_or_reg('2022-01-13', format, 'unix'))
                ,tooltip= {"placement": "top", "always_visible": True}
                ,marks= {int(to_unix_or_reg(day, the_format, 'unix')): {'label': day, 'style':{'color': electric1, 'fontWeight':'bold'}}  for day in days[::2]}
            )
            , dcc.Graph(
                id='slider-graph'
            )
        ]
        ,label="Conditional Probabilities"
    )
    ]
    )

]

    
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
@app.callback(
    Output('cond-prob-chart2', 'figure')
    ,Input("cond-prob-chart2-dccd", 'value')
)
def cond_prob_2(states):
    if not states:
        raise PreventUpdate
    df = df2_groups_pivoted_100_sorted_cut_imputed.loc[df2_groups_pivoted_100_sorted_cut_imputed['StateName'].isin(states)]
    fig = px.bar(
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
    fig.layout.legend.title = 'Rating'
    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
    ))
    return fig
@app.callback(
    Output('slider-graph', 'figure')
    ,Input('payment-lvl-slider', 'drag_value')
    ,Input('payment-lvl-slider', 'marks')
    ,Input('close-data-slider', 'drag_value')
    ,Input('close-data-slider', 'marks')
)
def plot_day_payment_lvl(lvl,lvl_marks, day, day_marks):
    # check out webgl
    if (not lvl) or (not day):
        raise PreventUpdate 
    new_day = to_unix_or_reg(the_date=day, the_format=the_format, option='unix_reg')
    lvl=lvl_marks[str(lvl)]['label']
    day_df = bus_cat_hold_loc_parquet.loc[(bus_cat_hold_loc_parquet['PaymentLevelName'] == lvl) & (bus_cat_hold_loc_parquet['CloseDate'] == new_day)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=day_df['EstimatedPopulation'], y=day_df['abs_review_diff'], mode='markers', marker_color='rgba(152, 0, 0, .8)', opacity=.50))

    fig.layout.xaxis.title = 'Estimated Population'
    fig.layout.yaxis.title = 'Absolute Review Difference'
    return fig


# if __name__ == '__main__':
# app.run_server(mode='jupyterlab')

app.run_server(mode='jupyterlab')





# showing multiple values on a scatter/line plot
states = ['Florida', 'Texas', 'Alaska']

state_date_df = bus_cat_hold_loc_parquet.groupby(['StateName', 'CloseDate'], as_index=False)['ReviewCount'].sum()
state_date_df_filter = state_date_df[state_date_df.StateName.isin(states)]
state_date_df_filter

fig = go.Figure()
fig.add_scatter(x=state_date_df_filter['CloseDate'], y=state_date_df_filter['ReviewCount'], mode='lines+markers+text')
fig.show()

# doing the same in plotly ex
fig = px.scatter(state_date_df_filter, x='CloseDate', y='ReviewCount', color='StateName')

figlines = px.line(state_date_df_filter, x='CloseDate', y='ReviewCount', color='StateName')

for trace in figlines.data:
    trace.showlegend = False
    fig.add_trace(trace)
fig.show()
