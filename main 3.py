# DEEP LEARNING COM TENSORFLOW
# PROBLEMA DE NEGÓCIO:

# Construir um modelo de Inteligência Artificial capaz de classificar imagens considerando 10 categorias: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
#'truck']. Dada uma nova imagem de uma dessas categorias o modelo deve ser capaz de classificar e indicar o que é a imagem.



# Imports
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# CARREGAMENTO DE DADOS

# Carrega o dataset CIFAR-10
(imagens_treino, labels_treino), (imagens_teste, labels_teste) = datasets.cifar10.load_data()

# Classes das imagens
nomes_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# PRÉ-PROCESSAMENTO E VISUALIZAÇÃO DAS IMAGENS

# Normaliza os valores dos pixels para que os dados fiquem na mesma escala
imagens_treino = imagens_treino / 255.0
imagens_teste = imagens_teste / 255.0

# Função para exibir as imagens
def visualiza_imagens(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])  # Removido cmap=plt.cm.binary para imagens coloridas
        plt.xlabel(nomes_classes[labels[i][0]])
    plt.show()

# Executa a função
visualiza_imagens(imagens_treino, labels_treino)

# CONSTRUÇÃO DO MODELO

# Cria o objeto de sequência de camadas
modelo_dsa = models.Sequential()

# Adiciona camadas convolucionais e de pooling
modelo_dsa.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
modelo_dsa.add(layers.MaxPooling2D((2, 2)))
modelo_dsa.add(layers.Conv2D(64, (3, 3), activation='relu'))
modelo_dsa.add(layers.MaxPooling2D((2, 2)))
modelo_dsa.add(layers.Conv2D(64, (3, 3), activation='relu'))
modelo_dsa.add(layers.MaxPooling2D((2, 2)))

# Adiciona camadas de classificação
modelo_dsa.add(layers.Flatten())
modelo_dsa.add(layers.Dense(64, activation='relu'))
modelo_dsa.add(layers.Dense(10, activation='softmax'))

# Exibe o sumário do modelo
modelo_dsa.summary()

# Compilação do modelo
modelo_dsa.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

# Treinamento do modelo
history = modelo_dsa.fit(imagens_treino,
                         labels_treino,
                         epochs=10,
                         validation_data=(imagens_teste, labels_teste))

# AVALIAÇÃO DO MODELO

# Avalia o modelo nos dados de teste
erro_teste, acc_teste = modelo_dsa.evaluate(imagens_teste, labels_teste, verbose=2)
print('\nAcurácia com Dados de Teste:', acc_teste)

# DEPLOY DO MODELO

# Carrega uma nova imagem
# Certifique-se de que o caminho da imagem está correto
caminho_imagem = "dados/nova_imagem.jpg"  # Altere para o caminho correto da sua imagem
try:
    nova_imagem = Image.open(caminho_imagem)
except FileNotFoundError:
    print(f"Erro: O arquivo '{caminho_imagem}' não foi encontrado. Verifique o caminho e tente novamente.")
    exit()

# Dimensões da imagem (em pixels)
print(nova_imagem.size)

# Obtém largura e altura da imagem
largura = nova_imagem.width
altura = nova_imagem.height

print("A largura da imagem é: ", largura)
print("A altura da imagem é: ", altura)

# Redimensiona para 32x32 pixels
nova_imagem = nova_imagem.resize((32, 32))

# Exibir a imagem
plt.figure(figsize=(1, 1))
plt.imshow(nova_imagem)
plt.xticks([])
plt.yticks([])
plt.show()

# Converte a imagem para um array NumPy e normaliza
nova_imagem_array = np.array(nova_imagem) / 255.0

# Expande a dimensão do array para que ele tenha o formato (1, 32, 32, 3)
nova_imagem_array = np.expand_dims(nova_imagem_array, axis=0)

# Previsões
previsoes = modelo_dsa.predict(nova_imagem_array)

# Obtém a classe com maior probabilidade e o nome da classe
classe_prevista = np.argmax(previsoes)
nome_classe_prevista = nomes_classes[classe_prevista]

print("A nova imagem foi classificada como:", nome_classe_prevista)




