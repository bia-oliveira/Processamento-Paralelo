BEATRIZ SANTOS DE OLIVEIRA, JULIA DE ARAÚJO RAMOS

Estrutura do projeto:

1. Pasta `DGEMM_Sequencial`
Conteúdo:
    `dgemm_sequencial.c`: Código-fonte da implementação sequencial.
    `resultados.csv`: Arquivo de saída gerado pelo programa, contendo o tempo de execução para cada tamanho de matriz.
Como Compilar e Executar:
     1.  Compilar: `gcc -o dgemm_sequencial dgemm_sequencial.c -O3`
     2.  Executar: `./dgemm_sequencial`

2. Pasta `DGEMM_Paralelo`
Conteúdo:
    `dgemm_paralelo.c`: Código-fonte da implementação paralela com diretivas OpenMP.
    `resultados2.csv`: Arquivo de saída gerado pelo programa, contendo o tempo de execução para cada combinação de tamanho de matriz e número de threads.
Como Compilar e Executar:
     1.  Compilar: `gcc -o dgemm_paralelo dgemm_paralelo.c -O3 -march=native -fopenmp`
     2.  Executar: `./dgemm_paralelo`

3. Pasta `Graficos`
Conteúdo:
    `graficos.py`: Script em Python para ler os arquivos de resultados e gerar os gráficos de análise.
    `resultados.csv` (entrada): Necessário para a execução do script.
    `resultados2.csv` (entrada): Necessário para a execução do script.
    Imagens (`.png`): Gráficos de tempo de execução e speedup gerados pelo script.
Como Executar:
     1.Pré-requisitos: Garanta que os arquivos `resultados.csv` e `resultados2.csv` estejam nesta pasta.
     2.Executar: `python graficos.py`
