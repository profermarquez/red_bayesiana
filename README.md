# contexto del problema:
Definimos la estructura de la red bayesiana, specificando las relaciones entre las variables meteorológicas.
Luego definir las Tablas de Probabilidad Condicional (CPTs), donde especificamos la probabilidad de cada nodo dado sus padres. Mas adelante entrenamos la red (opcional), con datos históricos, puodemos aprender las probabilidades condicionales. Al finalizar podemos realizar predicciones sobre eventos meteorológicos basados en evidencia observada.


# Crear y activar el entorno virtual
/virtualenv env         /env/Scripts/activate.bat

# requerimientos
pip install pgmpy networkx pandas numpy

pip install importlib-resources

pip install matplotlib

# ejecutar

py .\ejemplo_red.py  

# Libreria

https://pgmpy.org/


