BEATRIZ SANTOS DE OLIVEIRA, JULIA DE ARAÚJO RAMOS

Estrutura do projeto:

1. Pasta DGEMM_CUDA
    1.1 Conteúdo:
    dgemm_cuda.cu: Código-fonte unificado contendo as implementações sequencial, CUDA Naive e CUDA Tiled.
    resultados_cuda1.csv: Arquivo de saída da primeira execução.
    resultados_cuda2.csv: Arquivo de saída da segunda execução.
    resultados_cuda3.csv: Arquivo de saída da terceira execução.

    1.2 Como Compilar e Executar:
    Compilar: !nvcc -arch=sm_75 -o dgemm_cuda dgemm_cuda.cu -Xcompiler -fopenmp -O3
    Executar: ./dgemm_cuda (Execute 3 vezes renomeando os arquivos de saída para gerar os inputs necessários para os gráficos)


2. Pasta Graficos
   2.1 Conteúdo:
   graficos.py: Script em Python para ler os três arquivos de resultados, calcular as médias e gerar os gráficos comparativos.
   resultados_cuda1.csv (entrada): Resultado da execução 1.
   resultados_cuda2.csv (entrada): Resultado da execução 2.
   resultados_cuda3.csv (entrada): Resultado da execução 3.
   Imagens (.png): Gráficos gerados (Tempo, Speedup, Escalabilidade/GFLOPS).

   2.2 Como Executar:
   Pré-requisitos: Garanta que os três arquivos .csv gerados na etapa anterior estejam nesta pasta.
   Executar: python graficos.py

Esse projeto foi executado no wsl
