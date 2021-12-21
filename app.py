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
    '2.1. Método de Newton',
    '2.2. Método de Lagrange',
    '2.3. Método del Spline Cúbico',
    '2.4. Regresión Polinomial',
    '2.5. Comparación para 2021/09',
    '3. Predicción a corto plazo'])

st.sidebar.write('---------------------')

st.sidebar.write("""
# Autores
* P2 - Antonia Tapia
* P3 - Clemente Arévalo
* P3 - Leonardo Adriano""")

# =================================

# Datos y Funciones

data = pd.DataFrame({'Año': [0/12, 1/12, 2/12, 3/12, 4/12, 5/12, 6/12, 7/12, 8/12, 9/12, 10/12, 11/12, 12/12, 13/12, 14/12, 15/12, 16/12, 17/12, 18/12, 19/12, 20/12, 21/12, 22/12, 23/12, 24/12, 25/12, 26/12, 27/12, 28/12, 29/12, 30/12, 31/12, 32/12, 33/12, 34/12, 35/12, 36/12, 37/12, 38/12, 39/12, 40/12, 41/12, 42/12, 43/12, 44/12, 45/12, 46/12, 47/12, 48/12, 49/12, 50/12, 51/12, 52/12, 53/12, 54/12, 55/12, 56/12, 57/12, 58/12, 59/12, 60/12, 61/12, 62/12, 63/12, 64/12, 65/12, 66/12, 67/12, 68/12, 69/12, 70/12, 71/12, 72/12, 73/12, 74/12, 75/12, 76/12, 77/12, 78/12, 79/12, 80/12, 81/12, 82/12, 83/12, 84/12, 85/12, 86/12, 87/12, 88/12, 89/12, 90/12, 91/12],
                        'Regalías': [47800000, 45680657.03, 51614183.56, 46035855.65, 49174168.06, 49641024.43, 47053841.47, 43800679.3, 41190836, 36835599.42, 29702089.48, 21041758.77, 14202742.99, 15373293.74, 17561874.11, 17716471.52, 20956752.7, 18732231.87, 16719347.13, 14038974.51, 11595609.22, 10718626.47, 9259778.34, 7321209.43, 4716796.83, 4663126.4, 6148174.28, 7193858, 9058386.57, 8847384.49, 8901501.63, 9037362.57, 7280927.04, 8594233.01, 7185346.77, 10966335.68, 12215095.21, 10647723.58, 10026727.11, 10710810.43, 10453949.93, 8681822.47, 9531975.9, 10621617.31, 11755788.13, 13350898.68, 15663925.32, 16174379.98, 17310919.23, 14502808.77, 16769103.8, 18184774.26, 21684343.9, 20850902.4, 21139623.4, 21128436.06, 22304862.81, 24298087.6, 14941802.81, 15138566.04, 13708545.1, 16339759.71 ,16339759.71, 20086962.79, 21022026.49, 15584797.67, 16110381.89, 14959504.15, 16362664.56, 14272409.36, 16924300.06, 17009987.03, 16455603.74, 12806432.99, 5681552.81, 2407761.29, 3718328.31, 5529128.92, 6326405.52, 6088629.38, 5257140.57, 5622471.74, 5694707.13, 7073328.72, 8714281.41, 9631431.6, 11422757, 11067633.24, 13610720.3, 14551084.11, 15255383.4, 14215209.96]})
t = data['Año']
R = data['Regalías']
real = 15778928.50 / 1e6

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

    Los datos recolectados para este proyecto fueron extraídos de los Informes Mensuales de Actividades, emitidos por Perupetro.
    Encuentre los informes aquí: 
    
    Dirígase a uno de los siguientes apartados ubicados en la barra lateral izquierda:
    * **1. Visualización de Datos**: Se muestran una tabla con el historial de regalías y el gráfico correspondiente.
    * **2. Métodos de Interpolación y Regresión**: Se aplicarán los 4 métodos con el historial de regalías para su posterior comparación y elección de los métodos más precisos.
        * *2.1. Métodos de Newton*
        * *2.2. Métodos de Lagrange*
        * *2.3. Métodos del Spline Cúbico*
        * *2.4. Métodos de Regresión Polinomial*
        * *2.5. Comparación*
    * **3. Predicción a corto plazo**: Ejecute los métodos más precisos para un período de 4 meses y analice la predicción en el cambio de las regalías.""")

elif op == '1. Visualización de Datos':

    st.write('## **1. Datos**')

    st.write('#### Tabla del historial de regalías')
    st.write('Se considera al 2014 como año inicial o base, mientras que la información ha sido registrada mensualmente')
    st.dataframe(data)
    st.write('')
    st.write('')

    st.write('#### **Gráfica de la curva real del historial de regalías**')
    R11 = dict(x = t,y = R,mode = 'lines',type = 'scatter',name = 'Real',line = dict(color = 'rgb(26, 82, 118)', width= 3))
    rega = [R11]
    layout = go.Layout(xaxis = dict(title = 'Años'), yaxis = dict(title = 'Regalías Petroleras ($)'))
    figure = go.Figure(data=rega, layout=layout)
    st.write(figure)

elif op == '2.1. Método de Newton':

    st.write("""
    ## **2. Métodos de Interpolación y Regresión**
    ### **2.1. Método de Interpolación de Newton**
    """)

    n1 = st.slider('Elija hasta qué extremo desea observar la gráfica', min_value=1, max_value=8)
    x1 = np.linspace(0,n1,1000)
    R11 = dict(x = t,y = R,mode = 'lines',type = 'scatter',name = 'Real',line = dict(color = 'rgb(26, 82, 118)', width= 3))
    R12 = dict(x = x1,y = newton(t,R,x1),mode = 'lines',type = 'scatter',name = 'Método de Newton',line = dict(color = 'rgb(205, 12, 24)', width= 2, dash = 'dash'))
    rega1 = [R11,R12]

    layout = go.Layout(xaxis = dict(title = 'Años'), yaxis = dict(title = 'Regalías Petroleras ($)'))
    figure = go.Figure(data=rega1, layout=layout)
    figure.update_layout(yaxis_range=[0,max(R)])
    st.write(figure)

    st.write('-----------------------')

    val1 = newton(t,R,92/12)
    st.write('#### **Evaluación para el mes de setiembre del 2021**')
    st.write('* Predicción: ', '{:.2e}'.format(val1/1e6), 'M $')
    st.write('* Error Relativo: ', '{:.2e}'.format(abs(val1-real)/real * 100), ' %')

elif op == '2.2. Método de Lagrange':

    st.write("""
    ## **2. Métodos de Interpolación y Regresión**
    ### **2.2. Método de Interpolación de Lagrange**
    """)

    n2 = st.slider('Elija hasta qué extremo desea observar la gráfica', min_value=1, max_value=8)
    x2 = np.linspace(0,n2,1000)
    R11 = dict(x = t,y = R,mode = 'lines',type = 'scatter',name = 'Real',line = dict(color = 'rgb(26, 82, 118)', width= 3))
    R21 = dict(x = x2,y = laga(x2),mode = 'lines',type = 'scatter',name = 'Método de Lagrange',line = dict(color = 'rgb(46, 204, 113)', width= 2, dash = 'dash'))
    rega1 = [R11,R21]

    layout = go.Layout(xaxis = dict(title = 'Años'), yaxis = dict(title = 'Regalías Petroleras ($)'))
    figure = go.Figure(data=rega1, layout=layout)
    figure.update_layout(yaxis_range=[0,max(R)])
    st.write(figure)

    st.write('-----------------------')

    val2 = laga(92/12)
    st.write('#### **Evaluación para el mes de setiembre del 2021**')
    st.write('* Predicción: ', '{:.2e}'.format(val2/1e6), 'M $')
    st.write('* Error Relativo: ', '{:.2e}'.format(abs(val2-real)/real * 100), ' %')


elif op == '2.3. Método del Spline Cúbico':

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

        layout = go.Layout(xaxis = dict(title = 'Años'), yaxis = dict(title = 'Regalías Petroleras ($)'))
        figure = go.Figure(data=rega, layout=layout)
        st.write(figure)

    st.write('-----------------------')
    
    st.write("""
    #### **Evaluación para el mes de setiembre del 2021**  
      
    (Considere los nodos 4, 10, 24, 42, 54, 68, 75, 84 y 91)""")

    xm = [t[4], t[10], t[24], t[42], t[54], t[68], t[75], t[84], t[91]]
    ym = [R[4], R[10], R[24], R[42], R[54], R[68], R[75], R[84], R[91]]

    val3 = fspline(92/12)
    st.write('* Predicción: ', round(val3/1e6, 3), 'M $')
    st.write('* Error Relativo: ', round(abs(real-val3)/real * 100, 3), ' %')

    xsp = np.linspace(0,92/12,1000)
    L1 = dict(x = t,y = R,mode = 'lines',type = 'scatter',name = 'Real',line = dict(color = 'rgb(26, 82, 118)', width= 3))
    L2 = dict(x = xsp,y = fspline(xsp),mode = 'lines',type = 'scatter',name = 'Spline Cúbico',line = dict(color = 'rgb(205, 12, 24)', width= 2, dash = 'dash'))
    L3 = dict(x = xm,y = ym,mode = 'markers',type = 'scatter',name = 'Nodo del Spline',line = dict(color = 'rgb(0, 0, 0)'))
    rega = [L1,L2,L3]

    layout = go.Layout(xaxis = dict(title = 'Años'), yaxis = dict(title = 'Regalías Petroleras ($)'))
    figure = go.Figure(data=rega, layout=layout)
    figure.update_layout(xaxis_range=[0,92/12])
    st.write(figure)

elif op == '2.4. Regresión Polinomial':

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
    
    
    layout = go.Layout(xaxis = dict(title = 'Años'), yaxis = dict(title = 'Regalías Petroleras ($)'))
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
        st.write('* Polinomio de grado ', i+1, ': ', round(Rfif[i]/1e6, 3), ' M $')
        j = 0
    
    st.write('-------------------------')
    
    st.write('**Evaluación para el mes de setiembre del 2021**')
    real = 15778928.50
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

    valores = pd.DataFrame({'Grado del Polinomio': range(1,grad+1), 'Predicción (M $)': Pred, 'Error Relativo (%)': Error})
    st.dataframe(valores)

elif op == '2.5. Comparación para 2021/09':

    st.write("""
    ## **2. Métodos de Interpolación y Regresión**
    ### **2.4. Comparación para 2021/09**
    #### *Polinomios Interpoladores*
    """)

    val1 = newton(t,R,92/12) / 1e6
    val2 = laga(92/12) / 1e6
    xm = [t[4], t[10], t[24], t[42], t[54], t[68], t[75], t[84], t[91]]
    ym = [R[4], R[10], R[24], R[42], R[54], R[68], R[75], R[84], R[91]]
    val3 = fspline(92/12) / 1e6

    er1 = round(abs(val1 - real)/real * 100, 3)
    er2 = round(abs(val2 - real)/real * 100, 3)
    er3 = abs(val3 - real)/real * 100
    interp = pd.DataFrame({
        'Método': ['Newton', 'Lagrange', 'Spline'],
        'Predicción (M $)': [val1, val2, val3],
        'Error Relativo (%)': [er1, er2, er3]})
    st.dataframe(interp)

    st.write('#### *Polinomios de Regresión*')

    real = 15778928.50
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
        Error[i] = abs((Pred[i] - real)/real)
        Pred[i] = round(Pred[i]/1e6, 3)
        Error[i] = round(Error[i]*100, 3)
        j = 0
    
    regr = pd.DataFrame({
        'Grado': range(1,7),
        'Predicción (M $)': Pred,
        'Error Relativo (%)': Error})
    st.dataframe(regr)

elif op == '3. Predicción a corto plazo':

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
        'Spline Cúbico (M $)': splinefut,
        'Regresión de Grado 6 (M $)': polifut})
    st.dataframe(futuro)
    st.write('')
    st.write('')

    st.write('#### *Gráfico*')
    xsp = np.linspace(0,8,1000)
    tfut = np.linspace(0,8,97)
    

    Kr = dict(x = t,y = R,mode = 'lines',type = 'scatter',name = 'Real',line = dict(color = 'rgb(26, 82, 118)', width= 3))
    Ks1 = dict(x = xsp,y = fspline(xsp),mode = 'lines',type = 'scatter',name = 'Spline Cúbico',line = dict(color = 'rgb(205, 12, 24)', width= 2, dash = 'dash'))
    Ks2 = dict(x = xm,y = ym,mode = 'markers',type = 'scatter',name = 'Nodo del Spline',line = dict(color = 'rgb(0, 0, 0)'))
    Kr6 = dict(x = tfut,y = np.polyval(p6,tfut),mode = 'lines',type = 'scatter',name = 'Grado 6',line = dict(color = 'rgb(177, 26, 198)', width= 2, dash = 'dash'))
    rega = [Kr,Ks1,Ks2,Kr6]

    layout = go.Layout(xaxis = dict(title = 'Años'), yaxis = dict(title = 'Regalías Petroleras ($)'))
    figure = go.Figure(data=rega, layout=layout)
    st.write(figure)