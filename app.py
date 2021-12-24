import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.interpolate import *
from math import *

# =================================

st.set_page_config(layout="wide")

st.write('# Impacto del COVID-19 en las Regalías Petroleras del Perú')
st.write('---------------------')

st.sidebar.write("""
# XIV FERIA Y CONCURSO DE PROYECTOS VIRTUAL FIP 2021-2
* **Curso**: BEG01A - Economía General
* **Asesor**: PhD(c) Joseph Sinchitullo G.
* **Categoría**: C""")

st.sidebar.write('---------------------')

st.sidebar.write('# Navegación')
op = st.sidebar.radio('Seleccione un apartado:', 
    ['Información General',
    '1. Visualización de Datos',
    '2. Valores Nominales y Reales',
    '3.1. Método de Newton',
    '3.2. Método de Lagrange',
    '3.3. Método del Spline Cúbico',
    '3.4. Regresión Polinomial',
    '3.5. Comparación para 2021/09',
    '4. Predicción a corto plazo'])

st.sidebar.write('---------------------')

st.sidebar.write("""
# Autores
* P2 - Antonia Tapia
* P3 - Clemente Arévalo
* P3 - Leonardo Adriano""")

# =================================

# Datos y Funciones

data_inicial = pd.DataFrame({'Fecha': ['Ene-14', 'Feb-14', 'Mar-14', 'Abr-14', 'May-14', 'Jun-14', 'Jul-14', 'Ago-14', 'Set-14', 'Oct-14', 'Nov-14', 'Dic-14', 'Ene-15', 'Feb-15', 'Mar-15', 'Abr-15', 'May-15', 'Jun-15', 'Jul-15', 'Ago-15', 'Set-15', 'Oct-15', 'Nov-15', 'Dic-15', 'Ene-16', 'Feb-16', 'Mar-16', 'Abr-16', 'May-16', 'Jun-16', 'Jul-16', 'Ago-16', 'Set-16', 'Oct-16', 'Nov-16', 'Dic-16', 'Ene-17', 'Feb-17', 'Mar-17', 'Abr-17', 'May-17', 'Jun-17', 'Jul-17', 'Ago-17', 'Set-17', 'Oct-17', 'Nov-17', 'Dic-17', 'Ene-18', 'Feb-18', 'Mar-18', 'Abr-18', 'May-18', 'Jun-18', 'Jul-18', 'Ago-18', 'Set-18', 'Oct-18', 'Nov-18', 'Dic-18', 'Ene-19', 'Feb-19', 'Mar-19', 'Abr-19', 'May-19', 'Jun-19', 'Jul-19', 'Ago-19', 'Set-19', 'Oct-19', 'Nov-19', 'Dic-19', 'Ene-20', 'Feb-20', 'Mar-20', 'Abr-20', 'May-20', 'Jun-20', 'Jul-20', 'Ago-20', 'Set-20', 'Oct-20', 'Nov-20', 'Dic-20', 'Ene-21', 'Feb-21', 'Mar-21', 'Abr-21', 'May-21', 'Jun-21', 'Jul-21', 'Ago-21'],
                        'Año Relativo': [0/12, 1/12, 2/12, 3/12, 4/12, 5/12, 6/12, 7/12, 8/12, 9/12, 10/12, 11/12, 12/12, 13/12, 14/12, 15/12, 16/12, 17/12, 18/12, 19/12, 20/12, 21/12, 22/12, 23/12, 24/12, 25/12, 26/12, 27/12, 28/12, 29/12, 30/12, 31/12, 32/12, 33/12, 34/12, 35/12, 36/12, 37/12, 38/12, 39/12, 40/12, 41/12, 42/12, 43/12, 44/12, 45/12, 46/12, 47/12, 48/12, 49/12, 50/12, 51/12, 52/12, 53/12, 54/12, 55/12, 56/12, 57/12, 58/12, 59/12, 60/12, 61/12, 62/12, 63/12, 64/12, 65/12, 66/12, 67/12, 68/12, 69/12, 70/12, 71/12, 72/12, 73/12, 74/12, 75/12, 76/12, 77/12, 78/12, 79/12, 80/12, 81/12, 82/12, 83/12, 84/12, 85/12, 86/12, 87/12, 88/12, 89/12, 90/12, 91/12],
                        'Regalías Petroleras ($)': [47800000, 45680657.03, 51614183.56, 46035855.65, 49174168.06, 49641024.43, 47053841.47, 43800679.3, 41190836, 36835599.42, 29702089.48, 21041758.77, 14202742.99, 15373293.74, 17561874.11, 17716471.52, 20956752.7, 18732231.87, 16719347.13, 14038974.51, 11595609.22, 10718626.47, 9259778.34, 7321209.43, 4716796.83, 4663126.4, 6148174.28, 7193858, 9058386.57, 8847384.49, 8901501.63, 9037362.57, 7280927.04, 8594233.01, 7185346.77, 10966335.68, 12215095.21, 10647723.58, 10026727.11, 10710810.43, 10453949.93, 8681822.47, 9531975.9, 10621617.31, 11755788.13, 13350898.68, 15663925.32, 16174379.98, 17310919.23, 14502808.77, 16769103.8, 18184774.26, 21684343.9, 20850902.4, 21139623.4, 21128436.06, 22304862.81, 24298087.6, 14941802.81, 15138566.04, 13708545.1, 16339759.71 ,16339759.71, 20086962.79, 21022026.49, 15584797.67, 16110381.89, 14959504.15, 16362664.56, 14272409.36, 16924300.06, 17009987.03, 16455603.74, 12806432.99, 5681552.81, 2407761.29, 3718328.31, 5529128.92, 6326405.52, 6088629.38, 5257140.57, 5622471.74, 5694707.13, 7073328.72, 8714281.41, 9631431.6, 11422757, 11067633.24, 13610720.3, 14551084.11, 15255383.4, 14215209.96],
                        'IPC (2009 = 100)': [113.360708, 114.041823, 114.633197, 115.083960, 115.342934, 115.526534, 116.027125, 115.927783, 116.113850, 116.553757, 116.379555, 116.645938, 116.844581, 117.199168, 118.095348, 118.556518, 119.225610, 119.621846, 120.161085, 120.614364, 120.647579, 120.819863, 121.235570, 121.775943, 122.229583, 122.442374, 123.174724, 123.188774, 123.446933, 123.619152, 123.720207, 124.163479, 124.419832, 124.934127, 125.296516, 125.715251, 126.014256, 126.421498, 128.070740, 127.740249, 127.199476, 126.996976, 127.248793, 128.104192, 128.083990, 127.482962, 127.231518, 127.431083, 127.593452, 127.912717, 128.535811, 128.359623, 128.383312, 128.812187, 129.305262, 129.475644, 129.723825, 129.829788, 129.988749, 130.225039, 130.310118, 130.475301, 131.424577, 131.687532, 131.881921, 131.768144, 132.036077, 132.116589, 132.125022, 132.271318, 132.415375, 132.699434, 132.770837, 132.959600, 133.818519, 133.958470, 134.231951, 133.874764, 134.494187, 134.345922, 134.529108, 134.551676, 135.251990, 135.317902, 136.323181, 136.152415, 137.295372, 137.151635, 137.517223, 138.231861, 139.624601, 140.999813],
                        'Tipo de Cambio de $ a S/': [2.8092773, 2.8125500, 2.8065714, 2.7944400, 2.7869524, 2.7945238, 2.7863952, 2.8147667, 2.8646955, 2.9067091, 2.9255100, 2.9625381, 3.0067850, 3.0794350, 3.0922364, 3.1206400, 3.1513450, 3.1617810, 3.1819650, 3.2394429, 3.2186455, 3.2495200, 3.3382000, 3.3837762, 3.4386500, 3.5069571, 3.4069286, 3.3011000, 3.3346818, 3.3161000, 3.2994737, 3.3338273, 3.3824591, 3.3860190, 3.4034895, 3.3954238, 3.3398727, 3.2595750, 3.2639826, 3.2477556, 3.2734636, 3.2679750, 3.2493737, 3.2415591, 3.2465238, 3.2512955, 3.2407905, 3.2465000, 3.2149095, 3.2488100, 3.2522550, 3.2306850, 3.2741091, 3.2713450, 3.2772238, 3.2887190, 3.3118100, 3.3347682, 3.3756500, 3.3643842, 3.3438136, 3.3216000, 3.3043190, 3.3034050, 3.3335045, 3.3254750, 3.2904048, 3.3787100, 3.3571905, 3.3600762, 3.3727150, 3.3547381, 3.3277636, 3.3913800, 3.4925545, 3.3984200, 3.4218500, 3.4710952, 3.5173409, 3.5645476, 3.5557591, 3.5961364, 3.6087190, 3.6032429, 3.6249500, 3.6456900, 3.7091783, 3.6995250, 3.7747571, 3.9116048, 3.9424550, 4.0872524]})

data = pd.DataFrame({'Fecha': ['Ene-14', 'Feb-14', 'Mar-14', 'Abr-14', 'May-14', 'Jun-14', 'Jul-14', 'Ago-14', 'Set-14', 'Oct-14', 'Nov-14', 'Dic-14', 'Ene-15', 'Feb-15', 'Mar-15', 'Abr-15', 'May-15', 'Jun-15', 'Jul-15', 'Ago-15', 'Set-15', 'Oct-15', 'Nov-15', 'Dic-15', 'Ene-16', 'Feb-16', 'Mar-16', 'Abr-16', 'May-16', 'Jun-16', 'Jul-16', 'Ago-16', 'Set-16', 'Oct-16', 'Nov-16', 'Dic-16', 'Ene-17', 'Feb-17', 'Mar-17', 'Abr-17', 'May-17', 'Jun-17', 'Jul-17', 'Ago-17', 'Set-17', 'Oct-17', 'Nov-17', 'Dic-17', 'Ene-18', 'Feb-18', 'Mar-18', 'Abr-18', 'May-18', 'Jun-18', 'Jul-18', 'Ago-18', 'Set-18', 'Oct-18', 'Nov-18', 'Dic-18', 'Ene-19', 'Feb-19', 'Mar-19', 'Abr-19', 'May-19', 'Jun-19', 'Jul-19', 'Ago-19', 'Set-19', 'Oct-19', 'Nov-19', 'Dic-19', 'Ene-20', 'Feb-20', 'Mar-20', 'Abr-20', 'May-20', 'Jun-20', 'Jul-20', 'Ago-20', 'Set-20', 'Oct-20', 'Nov-20', 'Dic-20', 'Ene-21', 'Feb-21', 'Mar-21', 'Abr-21', 'May-21', 'Jun-21', 'Jul-21', 'Ago-21'],
                        'Año Relativo': [0/12, 1/12, 2/12, 3/12, 4/12, 5/12, 6/12, 7/12, 8/12, 9/12, 10/12, 11/12, 12/12, 13/12, 14/12, 15/12, 16/12, 17/12, 18/12, 19/12, 20/12, 21/12, 22/12, 23/12, 24/12, 25/12, 26/12, 27/12, 28/12, 29/12, 30/12, 31/12, 32/12, 33/12, 34/12, 35/12, 36/12, 37/12, 38/12, 39/12, 40/12, 41/12, 42/12, 43/12, 44/12, 45/12, 46/12, 47/12, 48/12, 49/12, 50/12, 51/12, 52/12, 53/12, 54/12, 55/12, 56/12, 57/12, 58/12, 59/12, 60/12, 61/12, 62/12, 63/12, 64/12, 65/12, 66/12, 67/12, 68/12, 69/12, 70/12, 71/12, 72/12, 73/12, 74/12, 75/12, 76/12, 77/12, 78/12, 79/12, 80/12, 81/12, 82/12, 83/12, 84/12, 85/12, 86/12, 87/12, 88/12, 89/12, 90/12, 91/12],
                        'Deflactor (2009 = 1)': [1.1336, 1.1404, 1.1463, 1.1508, 1.1534, 1.1553, 1.1603, 1.1593, 1.1611, 1.1655, 1.1638, 1.1665, 1.1684, 1.1720, 1.1810, 1.1856, 1.1923, 1.1962, 1.2016, 1.2061, 1.2065, 1.2082, 1.2124, 1.2178, 1.2223, 1.2244, 1.2317, 1.2319, 1.2345, 1.2362, 1.2372, 1.2416, 1.2442, 1.2493, 1.2530, 1.2572, 1.2601, 1.2642, 1.2807, 1.2774, 1.2720, 1.2700, 1.2725, 1.2810, 1.2808, 1.2748, 1.2723, 1.2743, 1.2759, 1.2791, 1.2854, 1.2836, 1.2838, 1.2881, 1.2931, 1.2948, 1.2972, 1.2983, 1.2999, 1.3023, 1.3031, 1.3048, 1.3142, 1.3169, 1.3188, 1.3177, 1.3204, 1.3212, 1.3213, 1.3227, 1.3242, 1.3270, 1.3277, 1.3296, 1.3382, 1.3396, 1.3423, 1.3387, 1.3449, 1.3435, 1.3453, 1.3455, 1.3525, 1.3532, 1.3632, 1.3615, 1.3730, 1.3715, 1.3752, 1.3823, 1.3962, 1.4100],
                        'Valor Nominal (S/)': [134283453.64, 128479131.93, 144858892.89, 128644436.46, 137046064.76, 138723024.70, 131110599.81, 123288692.07, 117999200.66, 107070371.70, 86893759.79, 62337011.95, 42704594.58, 47341058.81, 54305465.74, 55286729.68, 66041957.84, 59227213.92, 53200377.39, 45478455.70, 37322154.91, 34830391.09, 30910992.05, 24773334.15, 16219413.42, 16353384.44, 20946390.62, 23747644.64, 30206837.00, 29338811.71, 29370270.38, 30129005.81, 24627437.86, 29100236.67, 24455252.10, 37235357.27, 40796863.35, 34707053.59, 32727062.91, 34786094.08, 34220624.95, 28371978.79, 30972951.65, 34430600.15, 38165446.06, 43407716.19, 50763500.00, 52510124.61, 55653039.10, 47116870.16, 54537401.68, 58749277.43, 70996907.49, 68210495.31, 69279277.13, 69485490.12, 73869467.70, 81028489.41, 50438296.66, 50931952.55, 45838820.04, 54274145.85, 53991779.24, 66355373.32, 70077020.86, 51826855.03, 53009677.29, 50543826.27, 54932581.63, 47956382.87, 57080840.68, 57064051.49, 54760359.74, 43431480.71, 19843133.09, 8182584.12, 12723561.73, 19192133.07, 22252124.94, 21703209.36, 18693125.37, 20219175.08, 20550598.09, 25486921.19, 31588834.40, 35113213.87, 42369041.94, 40944985.86, 51377163.67, 56918089.90, 60143662.56, 58101150.75],
                        'Valor Real (S/)': [118456788.08, 112659661.65, 126367314.77, 111783115.96, 118816177.12, 120078928.97, 112999955.66, 106349564.26, 101623708.68, 91863509.56, 74664110.71, 53441219.66, 36548202.93, 40393681.64, 45984424.16, 46633226.60, 55392426.04, 49512038.06, 44274215.23, 37705671.36, 30934856.06, 28828364.99, 25496636.06, 20343372.87, 13269630.00, 13355984.45, 17005429.31, 19277442.15, 24469491.68, 23733225.18, 23739267.09, 24265594.08, 19793820.21, 23292464.09, 19517902.71, 29618806.77, 32374800.00, 27453442.76, 25553895.38, 27231897.82, 26903117.87, 22340672.73, 24340467.93, 26877028.47, 29797202.65, 34049817.73, 39898525.77, 41206684.72, 43617472.70, 36835172.66, 42429733.20, 45769281.69, 55300729.03, 52953448.66, 53578080.32, 53666842.64, 56943639.85, 62411323.82, 38802047.90, 39110721.68, 35176715.93, 41597256.67, 41081950.18, 50388500.95, 53136184.50, 39331854.77, 40147873.59, 38256986.99, 41576213.80, 36256070.93, 43107411.57, 43002482.96, 41244267.93, 32665171.01, 14828390.90, 6108299.18, 9478787.75, 14335885.64, 16545045.88, 16154721.36, 13895227.32, 15027070.40, 15194303.68, 18834848.02, 23172019.73, 25789637.20, 30859774.31, 29853808.06, 37360530.23, 41175811.05, 43075261.90, 41206544.55]})

t = data['Año Relativo']

Rega = data_inicial['Regalías Petroleras ($)']
IPC = data_inicial['IPC (2009 = 100)']
cambio = data_inicial['Tipo de Cambio de $ a S/']

D = data['Deflactor (2009 = 1)']
N = data['Valor Nominal (S/)']
R = data['Valor Real (S/)']
real = 45739396.18

def _poly_newton_coefficient(x, y):
    m = len(x)
    x = np.copy(x)
    a = np.copy(y)
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k - 1])/(x[k:m] - x[k - 1])
    return a

def newton(x_data, y_data, x):
    a = _poly_newton_coefficient(x_data, y_data)
    n = len(x_data) - 1  # Degree of polynomial
    p = a[n]
    for k in range(1, n + 1):
        p = a[n - k] + (x - x_data[n - k])*p
    return p

from scipy.interpolate import lagrange
laga = lagrange(t,R)

from scipy import interpolate
def fspline(tB):
    tck = interpolate.splrep(xm, ym)
    return interpolate.splev(tB, tck)

# =================================

if op == 'Información General':
    st.write("""
    # Información General
    Esta aplicación permite hacer un análisis gráfico de la tendencia histórica de la recaudación de las regalías así como su predicción a corto plazo por
    medio de polinomios interpoladores y de regresión.  

    Los datos recolectados para este proyecto fueron extraídos de:
    * Informes Mensuales de Actividades (Perupetro): https://bit.ly/3yXZUyR
    * Base de Datos de Estadísticas del BCRP (BCRP): https://www.bcrp.gob.pe/estadisticas.html

    Dirígase a uno de los siguientes apartados ubicados en la barra lateral izquierda:
    * **1. Visualización de Datos**: Se muestran una tabla con el historial de las regalías, del tipo de cambio y del IPC con sus gráficos correspondiente.
    * **2. Valores Reales y Nominales**: Se muestran las regalías en valores nominales y reales, considerando otras variables mostradas en los datos iniciales.
    * **3. Métodos de Interpolación y Regresión**: Se aplicarán los 4 métodos con el historial de regalías en valores reales para su posterior comparación y elección de los métodos más precisos.
        * *3.1. Métodos de Newton*
        * *3.2. Métodos de Lagrange*
        * *3.3. Métodos del Spline Cúbico*
        * *3.4. Métodos de Regresión Polinomial*
        * *3.5. Comparación*
    * **4. Predicción a corto plazo**: Ejecute los métodos más precisos para un período de 4 meses y analice la predicción en el cambio de las regalías.""")

elif op == '1. Visualización de Datos':

    st.write('## **1. Datos**')

    st.write('#### Registro de las regalías petroleras, el IPC y el tipo de cambio durante el período 2014-2021')
    st.write('Se considera al 2014 como año inicial o base en la recaudación. Toda la información ha sido registrada mensualmente')
    st.dataframe(data_inicial)
    st.write('---------------------')

    st.write('#### **Gráfico del historial de regalías (2014/01 - 2021/08)**')
    R11 = dict(x = t,y = Rega,mode = 'lines',type = 'scatter', line = dict(color = 'rgb(26, 82, 118)', width= 3))
    layout = go.Layout(xaxis = dict(title = 'Años'), yaxis = dict(title = 'Regalías Petroleras ($)'))
    figure = go.Figure(data=R11, layout=layout)
    st.write(figure)
    st.write('*Fuente*: Informes Mensuales de Actividades de Perupetro')
    st.write('---------------------')
    
    st.write('#### **Gráfico del IPC (2014/01 - 2021/08)**')
    R12 = dict(x = t,y = IPC,mode = 'lines',type = 'scatter', line = dict(color = 'rgb(192, 57, 43)', width= 3))
    layout = go.Layout(xaxis = dict(title = 'Años'), yaxis = dict(title = 'IPC (2009 = 100)'))
    figure = go.Figure(data=R12, layout=layout)
    st.write(figure)
    st.write('*Fuente*: Estadísticas del BCRP')
    st.write('---------------------')

    st.write('#### **Gráfico del Tipo de Cambio (2014/01 - 2021/08)**')
    R13 = dict(x = t,y = cambio,mode = 'lines',type = 'scatter', line = dict(color = 'rgb(25, 111, 61)', width= 3))
    layout = go.Layout(xaxis = dict(title = 'Años'), yaxis = dict(title = 'Tipo de Cambio de $ a S/'))
    figure = go.Figure(data=R13, layout=layout)
    st.write(figure)
    st.write('*Fuente*: Estadísticas del BCRP')

elif op == '2. Valores Nominales y Reales':

    st.write('## **2. Valores Nominales y Reales**')
    st.write('#### Deflactor, valores nominales y reales, en soles, respecto al año 2009 de las regalías petroleras')
    st.dataframe(data)
    st.write('---------------------')

    st.write('#### **Gráfico del Deflactor (2014/01 - 2021/08)**')
    R14 = dict(x = t,y = D,mode = 'lines',type = 'scatter', line = dict(color = 'rgb(186, 74, 0)', width= 3))
    layout = go.Layout(xaxis = dict(title = 'Años'), yaxis = dict(title = 'Deflactor'))
    figure = go.Figure(data=R14, layout=layout)
    st.write(figure)
    st.write('---------------------')
    
    st.write('#### **Gráfico de los Valores Nominales y Reales (2014/01 - 2021/08)**')
    R15 = dict(x = t,y = N,mode = 'lines',type = 'scatter',name = 'Valor Nominal',line = dict(color = 'rgb(212, 172, 13)', width= 2))
    R16 = dict(x = t,y = R,mode = 'lines',type = 'scatter',name = 'Valor Real',line = dict(color = 'rgb(26, 82, 118)', width= 3))
    ro = [R15,R16]
    layout = go.Layout(xaxis = dict(title = 'Años'), yaxis = dict(title = 'Valor (S/)'))
    figure = go.Figure(data=ro, layout=layout)
    st.write(figure)


elif op == '3.1. Método de Newton':

    st.write("""
    ## **2. Métodos de Interpolación y Regresión**
    ### **2.1. Método de Interpolación de Newton**
    """)

    n1 = st.slider('Elija hasta qué extremo desea observar la gráfica', min_value=1, max_value=8)
    x1 = np.linspace(0,n1,1000)
    R11 = dict(x = t,y = R,mode = 'lines',type = 'scatter',name = 'Real',line = dict(color = 'rgb(26, 82, 118)', width= 3))
    R12 = dict(x = x1,y = newton(t,R,x1),mode = 'lines',type = 'scatter',name = 'Método de Newton',line = dict(color = 'rgb(205, 12, 24)', width= 2, dash = 'dash'))
    rega1 = [R11,R12]

    layout = go.Layout(xaxis = dict(title = 'Años'), yaxis = dict(title = 'Regalías Petroleras Reales (S/)'))
    figure = go.Figure(data=rega1, layout=layout)
    figure.update_layout(yaxis_range=[0,max(R)])
    st.write(figure)

    st.write('-----------------------')

    val1 = newton(t,R,92/12)
    st.write('#### **Evaluación para el 2021/09**')
    st.write('* Predicción (S/): ', '{:.2e}'.format(val1/1e6), ' M')
    st.write('* Error Relativo (%): ', '{:.2e}'.format(abs(val1-real)/real * 100))

elif op == '3.2. Método de Lagrange':

    st.write("""
    ## **2. Métodos de Interpolación y Regresión**
    ### **2.2. Método de Interpolación de Lagrange**
    """)

    n2 = st.slider('Elija hasta qué extremo desea observar la gráfica', min_value=1, max_value=8)
    x2 = np.linspace(0,n2,1000)
    R11 = dict(x = t,y = R,mode = 'lines',type = 'scatter',name = 'Real',line = dict(color = 'rgb(26, 82, 118)', width= 3))
    R21 = dict(x = x2,y = laga(x2),mode = 'lines',type = 'scatter',name = 'Método de Lagrange',line = dict(color = 'rgb(46, 204, 113)', width= 2, dash = 'dash'))
    rega1 = [R11,R21]

    layout = go.Layout(xaxis = dict(title = 'Años'), yaxis = dict(title = 'Regalías Petroleras Reales (S/)'))
    figure = go.Figure(data=rega1, layout=layout)
    figure.update_layout(yaxis_range=[0,max(R)])
    st.write(figure)

    st.write('-----------------------')

    val2 = laga(92/12)
    st.write('#### **Evaluación para el 2021/09**')
    st.write('* Predicción (S/): ', '{:.2e}'.format(val2/1e6), ' M')
    st.write('* Error Relativo (%): ', '{:.2e}'.format(abs(val2-real)/real * 100))


elif op == '3.3. Método del Spline Cúbico':

    st.write("""
    ## **2. Métodos de Interpolación y Regresión**
    ### **2.3. Método del Spline Cúbico**
    """)
    
    nodos = st.multiselect('Seleccione la ubicación de los nodos en base a los índices mostrados en la tabla del apartado 1, elija por lo menos 4 nodos y de manera ascendente', range(0,92))
    
    if len(nodos) < 4:
        st.write('**Elija más nodos**')
        
    else:
        xm = [0] * len(nodos)
        ym = [0] * len(nodos)

        for i in range(len(nodos)):
            xm[i] = t[nodos[i]]
            ym[i] = R[nodos[i]]

        xsp = np.linspace(0,91/12,1000)
        L1 = dict(x = t,y = R,mode = 'lines',type = 'scatter',name = 'Real',line = dict(color = 'rgb(26, 82, 118)', width= 3))
        L2 = dict(x = xsp,y = fspline(xsp),mode = 'lines',type = 'scatter',name = 'Spline Cúbico',line = dict(color = 'rgb(205, 12, 24)', width= 2, dash = 'dash'))
        L3 = dict(x = xm,y = ym,mode = 'markers',type = 'scatter',name = 'Nodo del Spline',line = dict(color = 'rgb(0, 0, 0)'))
        rega = [L1,L2,L3]

        layout = go.Layout(xaxis = dict(title = 'Años'), yaxis = dict(title = 'Regalías Petroleras Reales (S/)'))
        figure = go.Figure(data=rega, layout=layout)
        st.write(figure)

    st.write('-----------------------')
    
    st.write("""
    #### **Evaluación para el 2021/09**  
      
    (Considere los nodos 4, 10, 24, 42, 54, 68, 75, 84 y 91)""")

    xm = [t[4], t[10], t[24], t[42], t[54], t[68], t[75], t[84], t[91]]
    ym = [R[4], R[10], R[24], R[42], R[54], R[68], R[75], R[84], R[91]]

    val3 = fspline(92/12)
    st.write('* Predicción (S/): ', round(val3/1e6, 3), ' M')
    st.write('* Error Relativo (%): ', round(abs(real-val3)/real * 100, 3))

    xsp = np.linspace(0,92/12,1000)
    L1 = dict(x = t,y = R,mode = 'lines',type = 'scatter',name = 'Real',line = dict(color = 'rgb(26, 82, 118)', width= 3))
    L2 = dict(x = xsp,y = fspline(xsp),mode = 'lines',type = 'scatter',name = 'Spline Cúbico',line = dict(color = 'rgb(205, 12, 24)', width= 2, dash = 'dash'))
    L3 = dict(x = xm,y = ym,mode = 'markers',type = 'scatter',name = 'Nodo del Spline',line = dict(color = 'rgb(0, 0, 0)'))
    rega = [L1,L2,L3]

    layout = go.Layout(xaxis = dict(title = 'Años'), yaxis = dict(title = 'Regalías Petroleras ($)'))
    figure = go.Figure(data=rega, layout=layout)
    figure.update_layout(xaxis_range=[0,92/12])
    st.write(figure)

elif op == '3.4. Regresión Polinomial':

    st.write("""
    ## **2. Métodos de Interpolación y Regresión**
    ### **2.4. Regresión Polinomial**
    """)

    grado = pd.DataFrame({'Grado': list(range(1,len(t)))})
    grad = st.selectbox('Seleccione hasta qué grado desea constuir polinomios de regresión', grado)
    
    p = [None] * grad
    Rp = [None] * (grad+1)
    Rp[0] = dict(x = t,y = R,mode = 'lines',type = 'scatter',name = 'Real',line = dict(color = 'rgb(26, 82, 118)', width= 3))

    for i in range(grad):
        p[i] = np.polyfit(t,R,i+1)

    for i in range(1,grad+1):
        Rp[i] = dict(x = t,y = np.polyval(p[i-1],t),mode = 'lines',type = 'scatter',name = f'Grado {i}',line = dict(width= 2, dash = 'dash'))
    
    
    layout = go.Layout(xaxis = dict(title = 'Años'), yaxis = dict(title = 'Regalías Petroleras Reales (S/)'))
    figure = go.Figure(data=Rp, layout=layout)
    st.write(figure)

    Rfif = [0] * grad

    tf = st.number_input('Ingrese un valor para calcular la predicción con todos los polinomios de regresión, considere solo números enteros')
    
    i = 0
    j = 0
    for i in range(grad):
        while len(p[i]) > j:
            Rfif[i] = Rfif[i] + (p[i][j] * (tf ** (len(p[i])-j-1)))
            j = j + 1
        st.write('* Polinomio de grado ', i+1, ' (S/) : ', round(Rfif[i]/1e6, 3), ' M')
        j = 0
    
    st.write('-------------------------')
    
    st.write('### **Evaluación para el 2021/09**')
    Pred = [0] * grad
    Error = [0] * grad
    i = 0
    j = 0
    for i in range(grad):
        while len(p[i]) > j:
            Pred[i] = Pred[i] + (p[i][j] * ((92/12) ** (len(p[i])-j-1)))
            j = j + 1
        Error[i] = abs((Pred[i] - real)/real)
        Pred[i] = round(Pred[i]/1e6, 3)
        Error[i] = round(Error[i]*100, 3)
        j = 0

    valores = pd.DataFrame({'Grado del Polinomio': range(1,grad+1), 'Predicción (M S/)': Pred, 'Error Relativo (%)': Error})
    st.dataframe(valores)

elif op == '3.5. Comparación para 2021/09':

    st.write("""
    ## **2. Métodos de Interpolación y Regresión**
    ### **2.4. Comparación para 2021/09**
    #### *Polinomios Interpoladores*
    """)
    real = 45739396.18
    valo1 = newton(t,R,92/12)
    valo2 = laga(92/12)
    xm = [t[4], t[10], t[24], t[42], t[54], t[68], t[75], t[84], t[91]]
    ym = [R[4], R[10], R[24], R[42], R[54], R[68], R[75], R[84], R[91]]
    valo3 = fspline(92/12)
    
    er1 = abs(valo1 - real)/real * 100
    er2 = abs(valo2 - real)/real * 100
    er3 = abs(valo3 - real)/real * 100
    interp = pd.DataFrame({
        'Método': ['Newton', 'Lagrange', 'Spline'],
        'Predicción (M S/)': [round(valo1/1e6, 3), round(valo2/1e6, 3), round(valo3/1e6, 3)],
        'Error Relativo (%)': [er1, er2, er3]})
    st.dataframe(interp)

    st.write('#### *Polinomios de Regresión*')

    p = [0] * 6
    for i in range(6):
        p[i] = np.polyfit(t,R,i+1)
    Pred = [0] * 6
    Error = [0] * 6
    i = 0
    j = 0
    for i in range(6):
        while len(p[i]) > j:
            Pred[i] = Pred[i] + (p[i][j] * ((92/12) ** (len(p[i])-j-1)))
            j = j + 1
        Error[i] = abs(Pred[i] - real)/real
        Pred[i] = round(Pred[i]/1e6, 3)
        Error[i] = round(Error[i]*100, 3)
        j = 0
    
    regr = pd.DataFrame({
        'Grado': range(1,7),
        'Predicción (M S/)': Pred,
        'Error Relativo (%)': Error})
    st.dataframe(regr)

elif op == '4. Predicción a corto plazo':

    xm = [t[4], t[10], t[24], t[42], t[54], t[68], t[75], t[84], t[91]]
    ym = [R[4], R[10], R[24], R[42], R[54], R[68], R[75], R[84], R[91]]
    splinefut = [0] * 4
    polifut = [0] * 4
    p6 = np.polyfit(t,R,6)

    for i in range(4):
        splinefut[i] = round(fspline((93 + i)/12) / 1e6, 2)
        polifut[i] = p6[0]*((93 + i)/12)**6 + p6[1]*((93 + i)/12)**5 + p6[2]*((93 + i)/12)**4 + p6[3]*((93 + i)/12)**3 + p6[4]*((93 + i)/12)**2 + p6[5]*((93 + i)/12)**1 + p6[6]
        polifut[i] = round(polifut[i]/1e6, 2)


    st.write('## **3. Predicción a corto plazo**')
    st.write('#### *Tabla de Predicción*')
    futuro = pd.DataFrame({
        'Fecha': ['2021/10', '2021/11', '2021/12', '2022/01'],
        'Spline Cúbico (M S/)': splinefut,
        'Regresión de Grado 6 (M $)': polifut})
    st.dataframe(futuro)
    st.write('')
    st.write('')

    st.write('#### *Gráfico de Predicción*')
    xsp = np.linspace(0,8,1000)
    tfut = np.linspace(0,8,97)
    

    Kr = dict(x = t,y = R,mode = 'lines',type = 'scatter',name = 'Real',line = dict(color = 'rgb(26, 82, 118)', width= 3))
    Ks1 = dict(x = xsp,y = fspline(xsp),mode = 'lines',type = 'scatter',name = 'Spline Cúbico',line = dict(color = 'rgb(205, 12, 24)', width= 2, dash = 'dash'))
    Ks2 = dict(x = xm,y = ym,mode = 'markers',type = 'scatter',name = 'Nodo del Spline',line = dict(color = 'rgb(0, 0, 0)'))
    Kr6 = dict(x = tfut,y = np.polyval(p6,tfut),mode = 'lines',type = 'scatter',name = 'Grado 6',line = dict(color = 'rgb(177, 26, 198)', width= 2, dash = 'dash'))
    rega = [Kr,Ks1,Ks2,Kr6]

    layout = go.Layout(xaxis = dict(title = 'Años'), yaxis = dict(title = 'Regalías Petroleras (S/)'))
    figure = go.Figure(data=rega, layout=layout)
    st.write(figure)