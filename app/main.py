import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
from jupyter_dash import JupyterDash
import dash_core_components as dcc
from dash.dependencies import Output, Input



# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app = JupyterDash(__name__)

app.layout = html.Div([
    html.H1('Gourmand Database',
        style={'color':'orange',
            'fontSize':'50px'}),
    html.H2('Businesses'),
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
            ),
    dcc.Dropdown(id = 'color_dropdown_menu', options=[
        {'label':color, 'value':color} for color in ['red', 'blue', 'green']
    ])
    ,html.Div(id='color_output')
]
)

@app.callback(Output('color_output', 'children'),
                Input('color_dropdown_menu', 'value'))
def display_desired_color(color):
    if not color:
        color = 'nothing'
    return f'Has eligido {color}'

if __name__ == '__main__':
    app.run_server(mode='inline')