# 💵 Contador Automático de Dinheiro com Visão Computacional
Este projeto foi desenvolvido como trabalho final da disciplina de visão computacional.
Nosso objetivo é implementar um sistema capaz de detectar e identificar moedas brasileiras a partir de imagens ou vídeos, e realizar automaticamente o somatório do valor total reconhecido.


## ⚙️ Tecnologias Utilizadas
O projeto foi construído em Python 3.10, utilizando as seguintes bibliotecas principais:
- OpenCV (opencv-python): Essencial para todo o processamento de imagem, incluindo a detecção de contornos das moedas e a exibição dos resultados visuais.
- TensorFlow/Keras: Utilizada para carregar nosso modelo de Inteligência Artificial (.h5) e executar a classificação de cada moeda identificada.
- NumPy: Usada para a manipulação eficiente das imagens como matrizes numéricas, sendo fundamental para a preparação dos dados antes de enviá-los ao modelo.
- [Google Teachable Machine](https://teachablemachine.withgoogle.com/): Plataforma online que usamos para treinar, de forma rápida e visual, o modelo de classificação de imagens.
- venv: Ferramenta para criar um ambiente virtual isolado, garantindo que as dependências do projeto sejam gerenciadas de forma limpa e sem conflitos.


## 📖 Como Funciona
O projeto opera em um pipeline de Visão Computacional e Machine Learning que pode ser dividido em quatro etapas principais:
1. Pré-processamento e Detecção de Contornos: primeiro, a imagem de entrada é carregada e passa por uma série de filtros (desfoque e detecção de bordas com Canny) para destacar claramente os contornos dos objetos. Em seguida, a função findContours do OpenCV é usada para identificar todas as formas fechadas, que são potenciais moedas.
2. Iteração e Recorte das Moedas: o sistema analisa cada contorno detectado. Contornos com área muito pequena são descartados para evitar ruído. Para cada contorno válido, uma _bounding box_ é calculada e usada para recortar a região exata da moeda da imagem original, isolando-a para a análise.
3. Classificação com o Modelo de IA: cada imagem recortada (contendo uma única moeda) é pré-processada (redimensionada para 224x224 e normalizada) e então enviada para o nosso modelo de classificação Keras. O modelo analisa a imagem e retorna a classe mais provável junto com um score de confiança.
4. Contabilização e Exibição do Resultado: se a confiança da classificação estiver acima de um determinado limite, a moeda é considerada válida. Seu valor correspondente é somado ao total, e um retângulo com a classe identificada é desenhado na imagem de saída. Ao final do processo, o valor total acumulado é exibido na tela para o usuário.


## 💻 Como executar
1. Clone o repositório
2. Crie e ative um ambiente virtual
```
python -m venv venv
source venv/bin/activate
```
3. Instale as dependências:
```
pip install -r requirements.txt
```
4. Ative o ambiente virtual:
```
venv/Scripts/activate
```
5. Execute o script principal:
```
python main.py
```


## Autores:
* [Alysson Antonietti](https://www.github.com/AlyssonAntonietti)
* [Gabriel Perico](https://github.com/GabrielPerico)
* [João Paulo Gregolon Paludo](https://github.com/joaopaludo)
