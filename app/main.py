import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
from jupyter_dash import JupyterDash
import dash_core_components as dcc
from dash.dependencies import Output, Input
import pandas as pd
from pathlib import Path
import os
path = Path(os.getcwd())
base_dir = path.parent
data_dir = os.path.join(base_dir, "data")
businesses = pd.read_csv(os.path.join(data_dir, "_Production_business.csv"), delimiter='|', quotechar="'")


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
# app = JupyterDash(__name__)

app.layout = html.Div([
    html.H1('Gourmand Database',
        style={'color':'orange',
            'fontSize':'50px'}),
    html.H2('Businesses'),
    dcc.Dropdown(id = 'chain_dropdown_menu', options=[
        {'label':chain , 'value':chain} for chain in businesses.iloc[:,2].unique()
    ])
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
    ])

    ,
    dbc.Row(
            [
                dbc.Col('BusinessID',width=1 ),
                dbc.Col('BusinessName', width =2)
                ]
            )
]
)

@app.callback(Output('chain_output', 'children'),
                Input('chain_dropdown_menu', 'value'))
def display_desired_chain_cnts(chain):
    if not chain:
        return f'{businesses.shape[0]} options to choose from'
    bus_df = businesses[businesses.iloc[:,2] == chain]
    chn_cnts = bus_df.shape[0]
    return [html.H3(chain), f'Has eligido {chain} y el numero de sus sucursales son {chn_cnts} ¿Quien lo iba a decir 😅?']


if __name__ == '__main__':
    app.run_server(mode='inline')