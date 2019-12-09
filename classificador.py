####     **************      Atenção as variaveis do sistema      **************
from keras.models import Sequential
from keras.layers import Dense
from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import time


# variaveis do sistema:
treinar = False             #pra não fazer o treinamento outra vez
# ler_pesos = True          #pra fazer o treinamento outra vez
# operacao = "getFrase"       #pega a frase e calcula quanto tempo levou pra digitar
operacao = "getTempo"     #recebe apenas o tempo e faz a predição


filename = "pesos-IA"
file = loadtxt('BaseDadosTreino.txt', delimiter=',')
x_treino = file[:, 0]
y_treino = file[:, 1]

file = loadtxt('BaseDadosTeste.txt', delimiter=',')
x_teste = file[:, 0]
y_teste = file[:, 1]

Classificador = Sequential()
                           # input_dim: Numero de entradas
Classificador.add(Dense(units = 1, activation='relu', input_dim=1)) #entrada
Classificador.add(Dense(units = 1, activation='relu')) #Layer escondida
Classificador.add(Dense(units = 1, activation='sigmoid')) #Saida

Classificador.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

if treinar == True:
    inicio = time.time() #marca o tempo inicial do treinamento
    hist = Classificador.fit( x_treino, y_treino, epochs=200, batch_size=25 ) # Treinamento
    fim = time.time() #marca o tempo final do treinamento
    print("Tempo: {} segundos".format(fim - inicio)  )

    Classificador.save_weights( filename ) #salva os pesos

    _, accuracy = Classificador.evaluate( x_treino, y_treino )  # Precisão
    print( 'Accuracy: %.2f' % (accuracy * 100) )  # imprime a eficiencia

    Classificador.save_weights( filename )  # Salva os pesos quando a IA estiver treinada

    # Plot
    plt.figure()
    plt.plot( hist.history['accuracy'], lw=2.0, color='b', label='train' )
    plt.title( 'Model accuracy' )
    plt.xlabel( 'Epochs' )
    plt.ylabel( 'Accuracy' )
    plt.legend( loc='upper right' )
    plt.show()


    plt.figure()
    plt.plot( hist.history['loss'], lw=2.0, color='b', label='train' )
    plt.title( 'Model Loss' )
    plt.xlabel( 'Epochs' )
    plt.ylabel( 'Loss' )
    plt.legend( loc='upper left' )
    plt.show()

else:
    Classificador.load_weights( './{}'.format( filename ) )




#resultados > 1 são valores 1, confirmando que está acima da idade
Resultado = Classificador.predict(x_teste)  #Testa os conjuntos de dados para teste e retorna a eficiencia
Resultado = np.ndarray.tolist(Resultado) #converte de ndarray para lista
map(lambda valor: valor[0] * 100, Resultado[0]) #transforma todos os resultados em porcentagem
print("Resultado: \n{}".format(Resultado))


# pega a frase e calcula quanto tempo levou pra digitar
while True and operacao == "getFrase":
    print("Escreva a frase:")
    print("\"    E como um anjo pendeu\n    As asas para voar...\n    Queria a lua do céu,\n    Queria a lua do mar...\"")
    print("Escreva>> ")
    inicio = time.time()
    frase = input()
    fim = time.time()
    tempo = fim - inicio

    if frase == 'exit':
        break

    sentence = str(tempo) + ",9"
    valor = StringIO( sentence )  # grava uma variavel como se fosse um arquivo de texto
    valor = loadtxt( valor, delimiter=',' )  # carrega um arquivo e devolve uma matriz numpy
    resultado = Classificador.predict( valor )

    if (np.argmax( resultado ) == 0):
        probabilidade = "%.5f%%" % (resultado[0] * 100)
        print( "Idoso => ", probabilidade )
    elif (np.argmax( resultado ) == 1):
        probabilidade = "%.5f%%" % (resultado[1] * 100)
        print( "idoso => ", probabilidade )
    print("*** Digitou em {} segundos ***".format(tempo))

# recebe apenas o tempo e faz a predição
while True and operacao == "getTempo":
    sentence = input( "input>> " )
    sentence = sentence + ",9" # 9 é um numero qualquer, apenas para ter duas dimensões
    if sentence == "exit,9":
        break

    valor = StringIO(sentence) #grava uma variavel como se fosse um arquivo de texto
    valor = loadtxt(valor, delimiter=',') #carrega um arquivo e devolve uma matriz numpy

    resultado = Classificador.predict( valor )

    if (np.argmax( resultado ) == 0):
        probabilidade = "%.5f%%" % (resultado[0] * 100)
        print( "Idoso => ", probabilidade )
    elif (np.argmax( resultado ) == 1):
        probabilidade = "%.5f%%" % (resultado[1] * 100)
        print( "idoso => ", probabilidade )

        # com 2 neuronios na camada escondida a rede só atingiu 51% de acuracia
        # eram 4000 epocas porem elas não influenciaram muito no desenpenho e a mesma acuracia se repetia até o fim das epocas
        # apenas gastava tempo com epocas desnecessarias

