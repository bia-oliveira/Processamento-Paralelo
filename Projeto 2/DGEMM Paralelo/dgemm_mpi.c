#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <math.h>  // Para fabs()
#include <string.h>  // Para memcpy()

// (a) Função DGEMM Sequencial (para validação em rank 0)
// Mesma lógica do código sequencial original
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

// (a) Função DGEMM Local (executada por cada processo MPI)
// Recebe apenas seu "pedaço" de A (A_local) e calcula seu "pedaço" de C (C_local)
void dgemm_local(double *A_local, double *B_global, double *C_local, int n, int rows_per_proc) {
    const int UNROLL = 4;
    int j;
    // O loop 'i' agora vai de 0 até rows_per_proc, pois é o número de linhas
    // que este processo recebeu.
    for (int i = 0; i < rows_per_proc; i++) {
        for (int k = 0; k < n; k++) {
            double valA = A_local[i * n + k]; // A_local tem 'rows_per_proc' linhas
            for (j = 0; j < n - (UNROLL - 1); j += UNROLL) {
                // C_local também tem 'rows_per_proc' linhas
                C_local[i * n + j]     += valA * B_global[k * n + j];
                C_local[i * n + j + 1] += valA * B_global[k * n + j + 1];
                C_local[i * n + j + 2] += valA * B_global[k * n + j + 2];
                C_local[i * n + j + 3] += valA * B_global[k * n + j + 3];
            }
            for (; j < n; j++) {
                C_local[i * n + j] += valA * B_global[k * n + j];
            }
        }
    }
}

// (b) Função de Validação
// Compara a matriz sequencial (C_seq) com a matriz montada do MPI (C_mpi)
double calcular_diferenca(double *C_seq, double *C_mpi, int n) {
    const double epsilon = 1e-12; // Conforme a imagem
    double max_diff = 0.0;
    long n_total = (long)n * n;

    for (long i = 0; i < n_total; i++) {
        double diff = fabs(C_seq[i] - C_mpi[i]);
        double rel_diff = diff / (fabs(C_seq[i]) + epsilon);
        if (rel_diff > max_diff) {
            max_diff = rel_diff;
        }
    }
    return max_diff;
}

// Função para inicializar uma matriz (usada apenas por rank 0)
void inicializar_matriz_aleatoria(double *matriz, int n) {
    for (int i = 0; i < n * n; i++) {
        matriz[i] = (double)rand() / (double)RAND_MAX;
    }
}

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int nprocs, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Rank 0 é responsável por E/S
    FILE *arquivo_saida = NULL;
    if (my_rank == 0) {
        arquivo_saida = fopen("resultados_mpi.csv", "a");
        if (arquivo_saida == NULL) {
            fprintf(stderr, "Erro ao abrir resultados_mpi.csv\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Escreve o cabeçalho se o arquivo estiver vazio
        fseek(arquivo_saida, 0, SEEK_END);
        if (ftell(arquivo_saida) == 0) {
            fprintf(arquivo_saida, "Tamanho,Processos,Tempo_s,Delta_Max,GFLOPS\n");
        }
    }

    int tamanhos[] = {512, 1024, 2048, 4096};
    int num_testes = sizeof(tamanhos) / sizeof(tamanhos[0]);

    if (my_rank == 0) {
        printf("Executando benchmarks MPI com %d processos...\n", nprocs);
    }

    for (int i = 0; i < num_testes; i++) {
        int n = tamanhos[i];

        // Validação da divisão de trabalho
        if (n % nprocs != 0) {
            if (my_rank == 0) {
                fprintf(stderr, "Tamanho %d não é divisível por %d processos. Pulando.\n", n, nprocs);
            }
            continue; // Pula este tamanho de matriz
        }

        int rows_per_proc = n / nprocs;
        long chunk_size = (long)rows_per_proc * n; // Tamanho do "pedaço" de A e C para cada processo

        // Alocação de Memória

        double *A_global = NULL, *B_global = NULL, *C_seq = NULL, *C_mpi = NULL;

        if (my_rank == 0) {
            // Rank 0 aloca as matrizes completas
            A_global = (double *)malloc((long)n * n * sizeof(double));
            B_global = (double *)malloc((long)n * n * sizeof(double));
            C_seq    = (double *)calloc((long)n * n, sizeof(double)); // Para validação
            C_mpi    = (double *)calloc((long)n * n, sizeof(double)); // Para resultado final

            // Inicializa dados
            srand(time(NULL));
            inicializar_matriz_aleatoria(A_global, n);
            inicializar_matriz_aleatoria(B_global, n);

            // (b) Rank 0 calcula o resultado sequencial ANTES de tudo para validação
            dgemm_seq(A_global, B_global, C_seq, n);

        } else {
            // Outros processos alocam espaço apenas para a matriz B global
            B_global = (double *)malloc((long)n * n * sizeof(double));
        }

        // Todos os processos alocam seus "pedaços" locais
        double *A_local = (double *)malloc(chunk_size * sizeof(double));
        double *C_local = (double *)calloc(chunk_size, sizeof(double));

        if (B_global == NULL || A_local == NULL || C_local == NULL || (my_rank == 0 && (A_global == NULL || C_seq == NULL || C_mpi == NULL))) {
            fprintf(stderr, "Rank %d: Erro de alocação de memória para n = %d\n", my_rank, n);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // (a) Distribuição de Dados 

        // 1. Transmite a matriz B inteira para todos os processos
        MPI_Bcast(B_global, (long)n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // 2. Distribui (Scatter) as linhas de A entre todos os processos
        MPI_Scatter(A_global, chunk_size, MPI_DOUBLE, 
                    A_local,  chunk_size, MPI_DOUBLE, 
                    0, MPI_COMM_WORLD);

        // (c) Medição de Desempenho 

        // Sincroniza todos os processos antes de iniciar o timer
        MPI_Barrier(MPI_COMM_WORLD); 
        double inicio = MPI_Wtime(); // Medição com MPI_Wtime()

        // (a) Computação Local
        dgemm_local(A_local, B_global, C_local, n, rows_per_proc);



        // (a) Coleta de Resultados

        // 3. Coleta (Gather) os pedaços C_local de volta para C_mpi no rank 0
        MPI_Gather(C_local,  chunk_size, MPI_DOUBLE, 
                   C_mpi,    chunk_size, MPI_DOUBLE, 
                   0, MPI_COMM_WORLD);

        double fim = MPI_Wtime();
        double tempo_execucao = fim - inicio;

        // (b) Validação e Saída (Apenas Rank 0) 

        if (my_rank == 0) {
            double delta = calcular_diferenca(C_seq, C_mpi, n);
            double gflops = (2.0 * n * n * n) / (tempo_execucao * 1e9);

            printf("  [N=%d, P=%d] Tempo: %.6f s, GFLOPS: %.2f, Delta: %e\n", 
                   n, nprocs, tempo_execucao, gflops, delta);

            fprintf(arquivo_saida, "%d,%d,%.6f,%.6e,%.6f\n", 
                    n, nprocs, tempo_execucao, delta, gflops);

            if (delta > 1e-9) {
                printf("  AVISO: Diferença (%.3e) maior que o limite (1e-9)!\n", delta);
            }
        }

        // Limpeza
        free(A_local);
        free(B_global); // Todos os processos alocaram e liberam B
        free(C_local);
        if (my_rank == 0) {
            free(A_global);
            free(C_seq);
            free(C_mpi);
        }
    } // Fim do loop de tamanhos

    if (my_rank == 0) {
        fclose(arquivo_saida);
        printf("Execução MPI concluída! Verifique 'resultados_mpi.csv'.\n");
    }

    MPI_Finalize();
    return 0;
}
