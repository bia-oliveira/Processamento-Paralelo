#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>  // Usando MPI para o timer
#include <math.h>

// Função para inicializar uma matriz com valores aleatórios
void inicializar_matriz_aleatoria(double *matriz, int n) {
    for (int i = 0; i < n * n; i++) {
        matriz[i] = (double)rand() / (double)RAND_MAX;
    }
}

// Função para multiplicação de matrizes (versão otimizada sequencial)
void dgemm_seq(double *A, double *B, double *C, int n) {
    const int UNROLL = 4; 
    int j;
    for (int i = 0; i < n; i++) { 
        for (int k = 0; k < n; k++) { 
            double valA = A[i * n + k]; 
            for (j = 0; j < n - (UNROLL - 1); j += UNROLL) { 
                C[i * n + j]     += valA * B[k * n + j];
                C[i * n + j + 1] += valA * B[k * n + j + 1];
                C[i * n + j + 2] += valA * B[k * n + j + 2];
                C[i * n + j + 3] += valA * B[k * n + j + 3];
            }
            for (; j < n; j++) { 
                C[i * n + j] += valA * B[k * n + j]; 
            }
        }
    }
}

// main agora inicializa MPI para usar o timer
int main(int argc, char **argv) {

    // Inicializa o ambiente MPI
    MPI_Init(&argc, &argv);

    // Este programa só deve ser executado por 1 processo, mas podemos
    // garantir que apenas o rank 0 escreva os arquivos.
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0) {
        srand(time(NULL));

        FILE *arquivo_saida = fopen("resultados_seq.csv", "w"); // "w" para criar um novo arquivo
        if (arquivo_saida == NULL) {
            fprintf(stderr, "Erro ao abrir o arquivo resultados_seq.csv\n");
            // Não use 'return' aqui, chame MPI_Abort
        }

        fprintf(arquivo_saida, "Tamanho,Tempo_s,GFLOPS\n");

        int tamanhos[] = {512, 1024, 2048, 4096};
        int num_testes = sizeof(tamanhos) / sizeof(tamanhos[0]);

        printf("Executando benchmark sequencial (com MPI_Wtime)...\n");

        for (int i = 0; i < num_testes; i++) {
            int n = tamanhos[i];
            printf("Testando N = %d...\n", n);

            double *A = (double *)malloc(n * n * sizeof(double));
            double *B = (double *)malloc(n * n * sizeof(double));
            double *C = (double *)calloc(n * n, sizeof(double));

            if (A == NULL || B == NULL || C == NULL) {
                fprintf(stderr, "Erro de alocação de memória para n = %d\n", n);
                free(A); free(B); free(C);
                continue;
            }

            inicializar_matriz_aleatoria(A, n);
            inicializar_matriz_aleatoria(B, n);

            // Medimos o tempo de execução com MPI_Wtime()
            double inicio = MPI_Wtime();
            dgemm_seq(A, B, C, n);
            double fim = MPI_Wtime();
            double tempo_execucao = fim - inicio;

            double gflops = (2.0 * n * n * n) / (tempo_execucao * 1e9);
            fprintf(arquivo_saida, "%d,%.6f,%.6f\n", n, tempo_execucao, gflops);

            free(A);
            free(B);
            free(C);
        }

        fclose(arquivo_saida);
        printf("Resultados sequenciais salvos em 'resultados_seq.csv'\n");
    }

    // Finaliza o ambiente MPI
    MPI_Finalize();
    return 0;
}
