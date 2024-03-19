import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

""" tabla de características de jugadores
    Habilidad técnica: control de pelota, dribles, acierto de pases, acierto de disparos
    Velocidad: Arranque aceleración, aceleración
    Fuerza: control de cuerpo frente a oponentes, dominio aereo
    Vision: lectura del campo, encontrar pases, vision espacial
    Resistencia: mantener resistencia fisica durante el partido

    Cada caracteristica tiene un puntaje del 1 al 5 dependiendo del tipo de jugador
"""

tipo_jugadores = {
"attacking_mid":[4,3,2,5,3],
"striker":[4,4,3,3,3],
"winger":[4,5,2,3,4],
"center_back":[2,3,5,3,4],
"fullback":[3,4,4,3,5]
}

#caracteristicas del jugador1
#attacking mid
#jugador1 = [4,2,2,5,3]

#fullback
jugador1 = [2,3,3,3,4]

#winger
#jugador1 = [4,5,2,3,4]

#striker
#jugador1 = [4,3,2,2,2]

#se definen a vectores de numpy 
tipo_jugadores_vectores = np.array(list(tipo_jugadores.values()))
jugador1_vector = np.array(jugador1)

#calcular los cosenos de similitud
similitudes = cosine_similarity(jugador1_vector.reshape(1,-1), tipo_jugadores_vectores)[0]

for tipo_jugador, similitud in zip(tipo_jugadores.keys(), similitudes):
    print(f"Calculo de {tipo_jugador}: {similitud}")

pos_recomendada = list(tipo_jugadores.keys())[np.argmax(similitudes)]
print(f"posicion recomendada es {pos_recomendada}")