import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Definir la estructura de la red
modelo = BayesianNetwork([
    ('Nube', 'Lluvia'),
    ('Nube', 'Temperatura'),
    ('Lluvia', 'Humedad'),
    ('Temperatura', 'Humedad')
])

# Visualizar la estructura con NetworkX
import matplotlib.pyplot as plt

G = nx.DiGraph()
G.add_edges_from(modelo.edges())
plt.figure(figsize=(6, 4))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=12)
plt.title("Estructura de la Red Bayesiana")
plt.show()

# Definir las CPTs
# Probabilidad de que haya nubes (a priori)
cpd_nube = TabularCPD(variable='Nube', variable_card=2, values=[[0.6], [0.4]])

# Probabilidad de lluvia dada la presencia de nubes
cpd_lluvia = TabularCPD(variable='Lluvia', variable_card=2,
                        values=[[0.8, 0.3], [0.2, 0.7]],
                        evidence=['Nube'], evidence_card=[2])

# Probabilidad de temperatura dada la presencia de nubes
cpd_temperatura = TabularCPD(variable='Temperatura', variable_card=3,
                             values=[[0.5, 0.2],  # Alta
                                     [0.3, 0.5],  # Media
                                     [0.2, 0.3]], # Baja
                             evidence=['Nube'], evidence_card=[2])

# Probabilidad de humedad dada la lluvia y la temperatura
cpd_humedad = TabularCPD(variable='Humedad', variable_card=3,
                          values=[[0.7, 0.4, 0.3, 0.8, 0.6, 0.2],  # Alta
                                  [0.2, 0.4, 0.5, 0.1, 0.3, 0.5],  # Media
                                  [0.1, 0.2, 0.2, 0.1, 0.1, 0.3]], # Baja
                          evidence=['Lluvia', 'Temperatura'], evidence_card=[2, 3])

# Agregar CPTs al modelo
modelo.add_cpds(cpd_nube, cpd_lluvia, cpd_temperatura, cpd_humedad)

# Verificar si el modelo es válido
print("¿El modelo es válido?", modelo.check_model())


# Inferencia
# Crear el objeto de inferencia
inferencia = VariableElimination(modelo)

# Calcular la probabilidad de lluvia dado que hay nubes
resultado = inferencia.query(variables=['Lluvia'], evidence={'Nube': 1})
print(resultado)

resultado_humedad = inferencia.query(variables=['Lluvia'], evidence={'Humedad': 0})
print(resultado_humedad)

# Aprender la estructura desde datos 
from pgmpy.estimators import BayesianEstimator, HillClimbSearch, BicScore

# Cargar datos (simulado)
datos = pd.DataFrame(np.random.randint(0, 2, size=(1000, 4)), columns=['Nube', 'Lluvia', 'Temperatura', 'Humedad'])

# Aprender estructura con búsqueda heurística
busqueda = HillClimbSearch(datos)
mejor_modelo = busqueda.estimate(scoring_method=BicScore(datos))

print("Estructura aprendida:", mejor_modelo.edges())

# Aprender CPTs a partir de los datos
modelo_aprendido = BayesianNetwork(mejor_modelo.edges())
modelo_aprendido.fit(datos, estimator=BayesianEstimator)
