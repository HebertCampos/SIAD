# perception_imovel_RNA
## SIAD - IFAL 2021.2


No trabalho você deve usar  RNA para prever resultados. Como exemplo mostrado em sala temos:

1. Previsão de preço de imóveis
    Dados Entrada: m2, bairro, valor do condomínio, etc
    Dados de Saída: valor do imóvel


    Como os valores de entrada e saída não estão entre 0 e 1 deve é necessário normatizar os valores de entrada e saída na etapa de treinamento e  desmoralizar os valores na etapa de previsão.
    Etapas:
    1. **Normalizar os valores**
        VALOR normalizado = (VALORnãonormalizado - MENORvalor) / (MAIORvalor - MENORvalor)
        MENORvalor menor valor dos exemplos usado no treinamento
        MAIORvalor  maior valor dos exemplos usado no treinamento
   
        **Para desnormalizar usa a mesma fórmula sendo que calcular o VALORnãonormalizado**.

    2. **Treinar a rede (significa calcular os pesos e bias)**. Deve-se parar quando o erro atingir um valor desejado ou atingir um número de repetições previamente estipulada.
    3. Salvar os pesos e bias da rede treinada
    4. Entrar com valores de entrada e verificar o valor de saída calculado pela RNA

Material extra de apoio: 

https://www.gsigma.ufsc.br/~popov/aulas/rna/neuronio_implementacao/

http://www.lps.usp.br/hae/apostila/redeneural.pdf

https://www.maxwell.vrac.puc-rio.br/32823/32823_4.PDF
