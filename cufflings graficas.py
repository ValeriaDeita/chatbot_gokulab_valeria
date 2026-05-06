import plotly.express as px
import pandas as pd

df = pd.DataFrame({
    'x': [1,2,3],
    'y': [4,5,6]
})

fig = px.line(df, x='x', y='y')
fig.show()