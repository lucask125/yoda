#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pickle
import random
import math
import random
import pandas as pd
import datetime as dt
import os

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # Initialize weights and biases for the layers
        self.weights = []
        self.biases = []

        # Initialize weights and biases for the first hidden layer
        self.weights.append(np.random.randn(input_size, hidden_sizes[0]))
        self.biases.append(np.zeros((1, hidden_sizes[0])))

        # Initialize weights and biases for additional hidden layers if present
        for i in range(1, len(hidden_sizes)):
            self.weights.append(np.random.randn(hidden_sizes[i-1], hidden_sizes[i]))
            self.biases.append(np.zeros((1, hidden_sizes[i])))

        # Initialize weights and biases for the output layer
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size))
        self.biases.append(np.zeros((1, output_size)))

    def forward(self, inputs):
        # Forward pass
        layer_output = inputs

        for i in range(len(self.weights) - 1):
            layer_activation = np.dot(layer_output, self.weights[i]) + self.biases[i]
            layer_output = self.sigmoid(layer_activation)

        output_activation = np.dot(layer_output, self.weights[-1]) + self.biases[-1]
        predicted_output = self.sigmoid(output_activation)*100000

        return predicted_output

    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def to_list(self):
        # Unpack weights and biases into a flat list
        flat_list = []
        for w, b in zip(self.weights, self.biases):
            flat_list.extend(w.flatten())
            flat_list.extend(b.flatten())
        return flat_list

    def from_list(self, flat_list):
        # Pack a flat list into weights and biases
        index = 0
        for i in range(len(self.weights)):
            # Calculate the size of the weight matrix
            weight_size = self.weights[i].size

            # Extract weights from the flat list
            self.weights[i] = np.array(flat_list[index:index + weight_size]).reshape(self.weights[i].shape)
            index += weight_size

            # Extract biases from the flat list
            bias_size = self.biases[i].size
            self.biases[i] = np.array(flat_list[index:index + bias_size]).reshape(self.biases[i].shape)
            index += bias_size

    def print_params(self):
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            print(f"Layer {i+1} - Weights:")
            print(w)
            print(f"Layer {i+1} - Biases:")
            print(b)
            print("\n")

    def print_params(self):
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            print(f"Layer {i+1}:")
            print(f"  Weights:")
            print(w)
            print(f"  Biases:")
            print(b)
            print("\n")

# Example usage
input_size = 1
hidden_sizes = [10, 6]  # You can customize the number of neurons in each hidden layer
output_size = 1

# Create a neural network
neural_network = NeuralNetwork(input_size, hidden_sizes, output_size)

# Sample input
example_input = np.random.randn(1, input_size)

# Make predictions
prediction = neural_network.forward(example_input)
print("Prediction:", prediction)

nn_size = len(NeuralNetwork(input_size, hidden_sizes, output_size).to_list())
print(nn_size)


# In[19]:


########### CLASSE DO INDIVÍDUO ###########

#cria a classe individuo, que carrega seu genoma, fitness e tipo de problema (se é rastrigin ou esfera)
class Individuo:
    # inicialização da classe
    def __init__(self, genoma):
        self.genoma = genoma
        self.rede_neural = NeuralNetwork(input_size, hidden_sizes, output_size)
        self.rede_neural.from_list(genoma)        
        self.calcular_fitness()

   # método para imprimir objetos
    def __str__(self):
        return "Genoma: " + str(self.genoma) + ' / Fitness: ' + str(round(self.fitness,2))
    
    # método para criar uma cópia do próprio objeto
    def copy(self):
        return Individuo(self.genoma)
    
    # a função fitness avalia simplesmente a saída da função dando o genoma como entrada
    def calcular_fitness(self):
        erro_quadratico_total = 0
        for x in range(100):
            erro_quadratico_total += ((x*x + 5) - self.rede_neural.forward(np.array([x]))[0])**2
        
        erro_quadratico_total = math.sqrt(erro_quadratico_total)/100
        
        # rodar a rede pra 100 valores
        #erro medio quadratico
        self.fitness = 100/(erro_quadratico_total+1)
    
    # aplica uma mutação no indivíduo
    def aplicar_mutacao(self,limite,chance):
        novo_genoma = self.genoma.copy()
        
        # altera um gene multiplicando-o por um fator
        if(random.random()>=chance):
            fator = random.uniform(0.5,2)
            gene = random.randint(0,len(self.genoma)-1)
            novo_gene = novo_genoma[gene]*fator
           
            # garantir que o gene não vai passar dos limites após a multiplicação
            if (novo_gene>limite[1]):
                novo_gene = limite[1]
            if (novo_gene < limite[0]):
                novo_gene = limite[0]
            
            novo_genoma[gene] = novo_gene
        
        # cria um gene completamente novo
        else:
            gene = random.randint(0,len(self.genoma)-1)
            novo_genoma[gene] = random.uniform(limite[0],limite[1])

        self.genoma = novo_genoma # atribui ao proprio individuo o novo genoma
        self.rede_neural.from_list(self.genoma)
        self.calcular_fitness() # recalcula sua fitness
    
    
########### CLASSE DO INDIVÍDUO ###########


# In[20]:


########### FUNÇÕES PARA O CÓDIGO PRINCIPAL ###########
# gera a população inicial
def gerar_populacao(npop,intervalo,n):

    populacao = []
    for i in range(0,npop):
        genoma = [] 
        for j in range(0,n):
            genoma.append(random.uniform(intervalo[0],intervalo[1])) #cria gene a gene e insere no indivíduo
        
        individuo = Individuo(genoma)
        # adiciona indivíduo à população
        populacao.append(individuo)
        
    return populacao

# função de recombinação
# utilizando crossover aritmético total
def recombinacao(individuos_pais):
    pais = []
    # extrai somente o genoma dos pais
    for pai in individuos_pais:
        pais.append(pai.genoma)
        
    alpha = random.uniform(0.1,0.9)
    n = len(pais[0]) # obtém o tamanho do vetor
    filhos = []
    
    for i in [(0,1),(1,0)]:
        filho = []
        for j in range(0,n):
            filho.append(alpha*pais[i[0]][j] + (1-alpha)*pais[i[1]][j])
        filhos.append(filho)
    
    individuos =[] #lista que vai carregar os individuos realmente e não somente o genoma
    for i in range(0,2):
        individuos.append(Individuo(filhos[i]))
    return(individuos)
########### FUNÇÕES PARA O CÓDIGO PRINCIPAL ###########


# In[21]:


########### CÓDIGO PRINCIPAL ###########
def save_results(file_path,resultado):
    # Open the file in binary write mode
    with open(file_path, 'wb') as file:
        # Use pickle.dump to save the variable to the file
        pickle.dump(resultado, file)
    
    print(f"Variable saved to {file_path}")
    
def rodar_otimizacao(n,npop,max_gen,intervalo_genoma,taxa_de_selecao,probabilidade_de_mutacao,chance_de_novo_gene,imprimir_resultados):
    # parâmetros iniciais
    #n -> quantidade de entradas da função
    #npop -> tamanho da população
    #max_gen -> limite de gerações 
    #intervalo_genoma -> são os limites do conjunto de entradas
    #taxa_de_selecao -> porcentagem da população que é selecionada para se reproduzir 
    #probabilidade_de_mutacao -> chance de ocorrer mutação em cada indivíduo por geração
    #chance_de_novo_gene -> caso haja mutação essa é a chance de surgir um gene totalmente novo
    folder_path = r'results\\' + dt.datetime.now().strftime('%d-%m-%Y %Hh%M')
    os.mkdir(folder_path)
    # geração de população inicial, o cálculo da fitness de cada indivíduo já é realizado na geração
    populacao = gerar_populacao(npop,intervalo_genoma,n)
    populacao.sort(key=lambda x: x.fitness, reverse=False) #ordena a população dos melhores indivíduos para os piores
    # variáveis de controle globais
    melhor_individuo_global = populacao[0]
    melhor_fitness_global = melhor_individuo_global.fitness
    # cria dataframe para guardar estatísticas da população
    estatisticas = pd.DataFrame()

    # loop principal, limitado por maxgen
    for i in range(0,max_gen):
        melhor_fitness= 0 # criado apenas para ser substituido pelo melhor da geração atual

        # SELEÇÃO
        # determina o numero de selecionados para reprodução
        numero_selecionados = int(taxa_de_selecao*npop)
        if(numero_selecionados % 2 != 0):
            numero_selecionados += 1

        lista_selecionados = []
        copia_populacao = populacao.copy()

        # cria a lista de selecionados considerando maior probabilidade para os de melhor fitness 
        while (len(lista_selecionados) < numero_selecionados):
            for j in range(0,len(copia_populacao)):
                if random.random() < (npop-j)/npop:
                    lista_selecionados.append(copia_populacao[j])

            # retira da lista de selecionáveis os que ja foram selecionados e roda novamente
            for j in lista_selecionados:
                copia_populacao.remove(j)

        # garante que o tamanho da lista dos que vao se reproduzir é o desejado
        lista_selecionados = lista_selecionados[0:numero_selecionados]

        # embaralha a lista
        random.shuffle(lista_selecionados)

        novos_individuos = []

        # RECOMBINAÇÃO
        # itera entre os selecionados e cria filhos para eles
        for j in range(0,int(numero_selecionados/2)):
            # pega os pais dois a dois
            pais = [lista_selecionados[j*2],lista_selecionados[(j*2)+1]]
            # coloca os novos individuos na lista de recombinados
            novos_individuos.extend(recombinacao(pais))

        # novos individuos sao adicionados na populacao e ordena novamente    
        populacao.extend(novos_individuos)
        populacao.sort(key=lambda x: x.fitness, reverse=True) #ordena a população dos melhores indivíduos para os piores

        # remove individuos "sobressalentes" da população baseado no fitness dos individuos, até que a população retorne ao número inicial
        while (len(populacao) > npop):
            individuo = random.choice(populacao)
            chance = populacao.index(individuo)/npop # em posicoes superiores a chance de ser removido é baixa, enquanto posicoes inferiores é alta
            if(random.random()<chance):
                populacao.remove(individuo)

        # MUTAÇÃO
        # itera por toda a população para aplicar mutação
        for individuo in populacao:
            if random.random() < probabilidade_de_mutacao:
                individuo.aplicar_mutacao(intervalo_genoma,chance_de_novo_gene) # aplica a mutação passando como parâmentro o intervalo máximo da população de entrada


        # ordena a população de acordo com fitness
        populacao.sort(key=lambda x: x.fitness, reverse=True) #ordena a população dos melhores indivíduos para os piores

        # determina melhor individuo da geração
        melhor_individuo = populacao[0].copy()
        melhor_fitness = melhor_individuo.fitness

        # calcula se há um novo melhor individuo global
        if(melhor_fitness > melhor_fitness_global):
            melhor_individuo_global = melhor_individuo.copy()
            melhor_fitness_global = melhor_individuo_global.fitness

        # calcula e salva as estatísticas
        media_fitness = 0
        for j in populacao:
            media_fitness += j.fitness

        media_fitness = media_fitness/len(populacao)

        new_row = pd.DataFrame({'Geração' : i,
                                'Melhor Fitness' : melhor_fitness,
                                'Melhor Fitness Global' : melhor_fitness_global,
                                'Média de Fitness' : media_fitness  }
                               ,index=[0])
        
        estatisticas = pd.concat([estatisticas,new_row],ignore_index=True)

        print('Geração ',i,'. Melhor fitness: ', round(melhor_fitness,3))
        
        save_results(folder_path+"\\"+"resultados_"+str(i)+".pkl",melhor_individuo)
        
        if round(melhor_fitness_global,3) == 100:
            break
    # FIM DO LOOP PRINCIPAL

    if(imprimir_resultados):
        print("Critério de convergência atingido na geração ",i)
        # imprime melhor individuo
        print('--------------------------------------------------------------------------')
        print("Melhor Indivíduo:")
        print(melhor_individuo_global)
        print('--------------------------------------------------------------------------')
        #print('População Final:')

        #imprime a população
        #for individuo in populacao:
        #    print(individuo)

        # IMPRIME AS ESTATÍSTICAS
        import matplotlib as plt

        plt.style.use('seaborn-v0_8')
        data = estatisticas

        ax = data.plot(x='Geração',y='Melhor Fitness',color ='green',style='.')
        data.plot(x='Geração',y='Melhor Fitness Global',color ='red',style='-.',ax=ax)
        data.plot(x='Geração',y='Média de Fitness',color ='blue',style='.',ax=ax)
        ax.legend()
        ax.set_ylabel('Fitness')
        string_titulo = 'Evolução da Otimização \n' + 'N = ' + str(n) + ' NPop = ' + str(npop) + ' MaxGen = ' + str(max_gen) + ' Sel = ' + str(taxa_de_selecao) + ' Mut = ' + str(probabilidade_de_mutacao)
        ax.set_title(string_titulo)
        # salvar a figura
        fig = ax.get_figure()
        fig.savefig(r'results\\' + dt.datetime.now().strftime('%d-%m-%Y %Hh%M') + str(n) + '-' + str(npop) + '-' + str(max_gen) + '-' + str(taxa_de_selecao) + '-' + str(probabilidade_de_mutacao)+ '.png', dpi=100)
    return populacao,[i,media_fitness,melhor_fitness_global]
    ########### CÓDIGO PRINCIPAL ###########


# In[22]:


print("-----------------------------------------Execução 1-----------------------------------------")
resultado,stats = rodar_otimizacao(nn_size,2500,5000,[-50,50],0.30,0.20,0.20,True)
# parâmetros iniciais
#n -> quantidade de entradas da função
#npop -> tamanho da população
#max_gen -> limite de gerações 
#intervalo_genoma -> são os limites do conjunto de entradas
#taxa_de_selecao -> porcentagem da população que é selecionada para se reproduzir 
#probabilidade_de_mutacao -> chance de ocorrer mutação em cada indivíduo por geração
#chance_de_novo_gene -> caso haja mutação essa é a chance de surgir um gene totalmente novo

# geração de população inicial, o cálculo da fitness de cada indivíduo já é realizado na geração

# Specify a file to save the variable



# In[ ]:


import matplotlib.pyplot as plt

# Example data
x = list(range(100))
y1 = []
y2 = []

for i in x:
    y1.append(i*i+5)
    y2.append(resultado[0].rede_neural.forward(np.array([i]))[0])

# Plotting y1 and y2 against x
plt.plot(x, y1, label='y1', marker='o', linestyle='-', color='b')
plt.plot(x, y2, label='y2', marker='s', linestyle='--', color='r')

# Adding labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot of y1 and y2 against x')

# Adding legend
plt.legend()

# Display the plot
plt.show()


# In[ ]:





# In[ ]:




