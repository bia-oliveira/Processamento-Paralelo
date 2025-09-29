#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// Função para inicializar uma matriz com valores aleatórios
void inicializar_matriz_aleatoria(double *matriz, int n) {
    for (int i = 0; i < n * n; i++) {
        matriz[i] = (double)rand() / (double)RAND_MAX;
    }
}

// Função para multiplicação de matrizes de forma paralela utilizando '#pragma omp parallel for'.
void dgemm(double *A, double *B, double *C, int n) {
    const int UNROLL = 4; // Unroll factor, faz quatro operações por iteração
    // Divide as iterações deste laço em blocos fixos, distribuindo um bloco para cada thread executar em paralelo.
    #pragma omp parallel for schedule(static) 
    for (int i = 0; i < n; i++) { // Itera sobre as linhas de A e C
        int j;
        for (int k = 0; k < n; k++) { // Itera sobre os elementos da linha i de A e da coluna k de B
            double valA = A[i * n + k]; // Pega o valor de A[i][k]
            // Transforma as operações em instruções vetorizadas e calcula quatro elementos por iteração de uma vez só
            #pragma omp simd
            for (j = 0; j < n - (UNROLL - 1); j += UNROLL) { // Faz o somatório da multiplicação de cada elemento da linha i de A com a coluna k de B
                C[i * n + j]     += valA * B[k * n + j];
                C[i * n + j + 1] += valA * B[k * n + j + 1];
                C[i * n + j + 2] += valA * B[k * n + j + 2];
                C[i * n + j + 3] += valA * B[k * n + j + 3];
            }
            for (; j < n; j++) { // Para o caso de n não ser múltiplo de UNROLL
                C[i * n + j] += valA * B[k * n + j]; // Multiplica e acumula o valor que sobrou 
            }
        }
    }
}

int main() {

    // Inicializa a semente do gerador de números aleatórios
    srand(time(NULL));

    // Abre o arquivo de saída para escrita
    FILE *arquivo_saida = fopen("resultados2.csv", "a");
    if (arquivo_saida == NULL) {
        fprintf(stderr, "Erro ao abrir o arquivo de saida\n");
        return 1;
    }

    // Escreve o cabeçalho apenas uma vez
    fseek(arquivo_saida, 0, SEEK_END);
    long tamanho_arquivo = ftell(arquivo_saida);
    if (tamanho_arquivo == 0) {
        fprintf(arquivo_saida, "Tamanho,Threads,Tempo_s,GFLOPS\n");
    }

    // Define os tamanhos das matrizes e o número de threads a serem testados
    int tamanhos[] = {512, 1024, 2048, 4096};
    int num_testes = sizeof(tamanhos) / sizeof(tamanhos[0]);
    int threads_para_testar[] = {2, 4, 6, 8, 10, 12};
    int num_configs_thread = sizeof(threads_para_testar) / sizeof(threads_para_testar[0]);

    printf("Executando benchmarks...\n");

    // Loop para testar diferentes números de threads
    for (int t = 0; t < num_configs_thread; t++) {

        printf("Rodando com %d threads...\n", threads_para_testar[t]);

        
        omp_set_dynamic(0); // Controle de threads dinâmico desativado
        omp_set_num_threads(threads_para_testar[t]); // Pega o número de threads a serem usadas

        // Loop para testar diferentes tamanhos de matriz
        for (int i = 0; i < num_testes; i++) {
            
            int n = tamanhos[i]; // Pega o tamanho da matriz

            // Aloca memória para as matrizes A, B e C em vetores unidimensionais
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

            // Usamos omp_get_wtime() para medir o tempo real 
            double inicio = omp_get_wtime();
            dgemm(A, B, C, n);
            double fim = omp_get_wtime();

            double tempo_execucao = fim - inicio;

            fprintf(arquivo_saida, "%d,%d,%.6f\n", n, threads_para_testar[t], tempo_execucao);

            free(A);
            free(B);
            free(C);
        }
    }

    fclose(arquivo_saida);
    printf("Execucao concluida! Verifique 'resultados2.csv'.\n");

    return 0;
}
