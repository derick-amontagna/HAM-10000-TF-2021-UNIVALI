**Sumário**

[TOCM]

[TOC]

#Dados
+ Dados de treinamento: Os dados e tabela verdade utilizados nessa etapa podem ser encontrados no [Kaggle Datasets](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000);
+ Dados de validação e teste: Os dados utilizados nessa etapa podem ser baixado respectivamente em:  [ISIC 2018 - Changelle 3 - Validação - Dados](https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Validation_Input.zip) e [ISIC 2018 - Changelle 3 - Validação - Tabela Verdade](https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Validation_GroundTruth.zip); [ISIC 2018 - Changelle 3 - Teste - Dados](https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Test_Input.zip), até esse momento não a uma tabela verdade para o conjunto de teste, para verificar as metricas de avaliação, o resultado pode ser verificado [ISIC 2018 - Changelle 3 - Teste - Verificar das metricas](89N3PDyZzakoH7W6n8ZrjGDDktjh8iWFG6eKRvi3kvpQ).

Qualquer outro conjunto de dados das demais competições do ISIC podem ser visto  [aqui](https://challenge.isic-archive.com/data/).

#Pré-processamento
+ 1) Definição dos Folders para a execução do método *K-Fold Cross-validation ( KFCV)* utilizando o arquivo CSV de metadados do HAM10000;
+ 2) Aplicação do algoritmo de constância de cor em todo o conjunto de dados do HAM10000;
+ 3) Geração dos Folders que serão utilizados no treinamento para a aplicação do método KFCV;
+ 4) Aplicação do algoritmo de constância de cor em todo o conjunto de dados de validação e teste.

#Treinamento
Após a etapa de pré-processamento os dados podem ser agora utilizado para a etapa de treinamento. De maneira, que é o código e os dados utilizados podem ser encontrado dentro do [Kaggle Code - Treinamento](https://www.kaggle.com/code/derickabreumontagna/ttciii-derick-treinamento-ham10000/notebook).

# Validação e teste
Assim como o treinamento, os códigos e dados utilizados nas etapas de validação e teste podem ser encontrados em: [Kaggle Code -  Validação e Teste](https://www.kaggle.com/code/derickabreumontagna/ttciii-derick-valida-oiteste-ham10000/notebook).

# Documentos gerados
Os detalhes a cerca de cada etapa e explicações podem ser lidas a partir dos seguintes:
## TTTC
Esse é o trabalho completo:

## Artigo

