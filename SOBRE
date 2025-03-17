🧠 Projeto: Classificação de Imagens com CNN e TensorFlow
🎯 Objetivo do Projeto
O objetivo deste projeto é desenvolver uma rede neural convolucional (CNN) utilizando TensorFlow e Keras, capaz de classificar imagens em 10 categorias diferentes:

✈️ airplane
🚗 automobile
🐦 bird
🐱 cat
🦌 deer
🐶 dog
🐸 frog
🐴 horse
🚢 ship
🚚 truck
Dada uma nova imagem de uma dessas categorias, o modelo deve ser capaz de reconhecer e classificá-la corretamente.

🔧 Etapas do Projeto
1️⃣ Carregamento e Pré-processamento dos Dados

Dataset: O projeto usa o CIFAR-10, que contém 60.000 imagens coloridas (32x32 pixels) divididas em 10 classes.
Normalização: Os pixels das imagens são normalizados para ficar entre 0 e 1, facilitando o aprendizado da rede.
2️⃣ Exploração e Visualização dos Dados

Exibe exemplos das imagens para garantir que o conjunto de dados foi carregado corretamente.
3️⃣ Criação da Rede Neural Convolucional (CNN)
A arquitetura da CNN contém:

3 camadas convolucionais com ReLU para detectar padrões como bordas e texturas.
3 camadas de pooling (MaxPooling) para reduzir a dimensionalidade e evitar overfitting.
Camada Flatten para achatar a saída das camadas convolucionais em um vetor.
Camada densa (fully connected) intermediária com 64 neurônios e ativação ReLU.
Camada de saída com 10 neurônios e ativação Softmax (para classificar entre as 10 categorias).
4️⃣ Compilação e Treinamento do Modelo

Otimizador: Adam (eficiente para grandes volumes de dados).
Função de perda: Sparse Categorical Crossentropy (ideal para múltiplas classes).
Métrica: Acurácia.
Treinamento: 10 épocas, validando o desempenho com os dados de teste.
5️⃣ Avaliação do Modelo

Mede a acurácia do modelo nos dados de teste para garantir que ele generaliza bem.
6️⃣ Deploy e Classificação de uma Nova Imagem

O modelo é testado com uma imagem externa carregada, redimensionada e normalizada.
A imagem é então classificada e o modelo retorna o nome da categoria prevista.
📈 Resultados Esperados
Acurácia satisfatória para um modelo inicial (normalmente entre 70% e 80% com CIFAR-10).
Capacidade de reconhecer uma nova imagem corretamente entre as 10 categorias aprendidas.
Código modular e fácil de adaptar para outros datasets de classificação de imagens.
🔥 Possíveis Melhorias Futuras
🔹 Aumentar o número de épocas ou ajustar hiperparâmetros (learning rate, batch size, etc.).
🔹 Implementar técnicas de data augmentation para melhorar a capacidade de generalização.
🔹 Salvar o modelo treinado e criar uma interface gráfica para upload de imagens.
🔹 Usar redes mais avançadas, como ResNet ou MobileNet, para ganhar performance.
