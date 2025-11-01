BEATRIZ SANTOS DE OLIVEIRA, JULIA DE ARAÚJO RAMOS

Este projeto implementa e avalia a multiplicação de matrizes (DGEMM) usando o padrão de memória distribuída (MPI), os resultados são comparados com uma implementação sequencial de referência. O fluxo de trabalho e os comandos abaixo são destinados a um ambiente WSL (Ubuntu).

PRÉ-REQUISITOS (Ambiente WSL): Antes de compilar, instale as ferramentas necessárias (GCC, MPICH) no terminal WSL:

1. Atualize os pacotes:
   sudo apt update

2. Instale o compilador C e o MPI:
   sudo apt install build-essential mpich


Estrutura do projeto:

1. Pasta `DGEMM Sequencial`
Conteúdo:
    `dgemm_seq.c`: Código-fonte da implementação sequencial. Utiliza MPI_Wtime() para uma medição de tempo justa.
    `resultados_seq.csv`: Arquivo de saída gerado pelo programa, contendo o tempo de execução e GFLOPS para cada tamanho de matriz.

Como Compilar e Executar:
    1.  Compilar: `mpicc -o dgemm_seq dgemm_seq.c -O3 -lm`
    2.  Executar (3 vezes para a média):
        mpiexec ./dgemm_seq

2. Pasta `DGEMM Paralelo`
Conteúdo:
    `dgemm_mpi.c`: Código-fonte da implementação paralela com MPI.
    `resultados_mpi.csv`: Arquivo de saída gerado, contendo tempo, delta e GFLOPS para cada combinação de tamanho e número de processos.

Como Compilar e Executar:
    1.  Compilar: `mpicc -o dgemm_mpi dgemm_mpi.c -O3 -lm`
    2.  Executar (3 vezes para CADA contagem de processo):

        # Para 1 processo
        mpiexec -n 1 ./dgemm_mpi
        mpiexec -n 1 ./dgemm_mpi
        mpiexec -n 1 ./dgemm_mpi

        # Para 2 processos
        mpiexec -n 2 ./dgemm_mpi
        mpiexec -n 2 ./dgemm_mpi
        mpiexec -n 2 ./dgemm_mpi

        # Para 4 processos
        mpiexec -n 4 ./dgemm_mpi
        mpiexec -n 4 ./dgemm_mpi
        mpiexec -n 4 ./dgemm_mpi

        # Para 8 processos
        mpiexec -n 8 ./dgemm_mpi
        mpiexec -n 8 ./dgemm_mpi
        mpiexec -n 8 ./dgemm_mpi

3. Pasta `Graficos`
Conteúdo:
    `graficos.py`: Script em Python para ler os arquivos de resultados e gerar os gráficos de análise (Tempo, Speedup e Eficiência).
    `resultados_seq.csv` (entrada): Necessário para a execução do script.
    `resultados_mpi.csv` (entrada): Necessário para a execução do script.
    Imagens (`.png`): Gráficos de saída gerados pelo script.

Como Executar:
    1. Pré-requisitos:
       - Garanta que os arquivos `resultados_seq.csv` e `resultados_mpi.csv` estejam nesta pasta.
    2. Executar: `python analise_mpi.py`
