/**
 * Projeto 3: Multiplicação de Matrizes (DGEMM) com CUDA
 * Disciplina: Processamento Paralelo (DEC107)
 * Discentes: Beatriz Santos de Oliveira e Júlia de Araújo Ramos
 *
 * Compilação: !nvcc -arch=sm_75 -o dgemm_cuda dgemm_cuda.cu -Xcompiler -fopenmp -O3
 * Execução: ./dgemm_cuda
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32 // Tamanho do bloco para a versão Tiled

// FUNÇÕES AUXILIARES

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Erro CUDA em %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void inicializar_matriz_aleatoria(double *matriz, int n) {
    for (int i = 0; i < n * n; i++) {
        matriz[i] = (double)rand() / (double)RAND_MAX;
    }
}

// Validação: Calcula a diferença relativa máxima
double calcular_delta(double *C_seq, double *C_cuda, int n) {
    double max_diff = 0.0;
    double epsilon = 1e-12;
    for (long i = 0; i < (long)n * n; i++) {
        double diff = fabs(C_seq[i] - C_cuda[i]);
        double rel_diff = diff / (fabs(C_seq[i]) + epsilon);
        if (rel_diff > max_diff) max_diff = rel_diff;
    }
    return max_diff;
}

// 1. VERSÃO SEQUENCIAL (Referência)
// Mesma lógica do código anterior (Unroll 4)
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

// 2. KERNEL CUDA NAIVE (Memória Global)
__global__ void dgemm_naive_kernel(double *A, double *B, double *C, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// 3. KERNEL CUDA TILED (Memória Compartilhada)
__global__ void dgemm_tiled_kernel(double *A, double *B, double *C, int n) {
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    double sum = 0.0;

    for (int m = 0; m < (n + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        if (row < n && (m * TILE_SIZE + tx) < n)
            As[ty][tx] = A[row * n + (m * TILE_SIZE + tx)];
        else
            As[ty][tx] = 0.0;

        if (col < n && (m * TILE_SIZE + ty) < n)
            Bs[ty][tx] = B[(m * TILE_SIZE + ty) * n + col];
        else
            Bs[ty][tx] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

// MAIN
int main() {
    srand(time(NULL));

    int tamanhos[] = {512, 1024, 2048, 4096};
    int num_testes = sizeof(tamanhos) / sizeof(int);

    FILE *f_out = fopen("resultados_cuda.csv", "w");
    if (!f_out) { printf("Erro ao criar arquivo.\n"); return 1; }
    fprintf(f_out, "Tamanho,Metodo,Tempo_s,GFLOPS,Delta\n");

    printf("Iniciando Benchmark CUDA...\n");

    for (int t = 0; t < num_testes; t++) {
        int n = tamanhos[t];
        size_t bytes = (size_t)n * n * sizeof(double);
        printf("\ Matriz %dx%d\n", n, n);

        // Alocação no Host
        double *h_A = (double*)malloc(bytes);
        double *h_B = (double*)malloc(bytes);
        double *h_C_seq = (double*)malloc(bytes);
        double *h_C_cuda = (double*)malloc(bytes);

        if (!h_A || !h_B || !h_C_seq || !h_C_cuda) {
            fprintf(stderr, "Erro de alocacao no host.\n"); exit(1);
        }

        inicializar_matriz_aleatoria(h_A, n);
        inicializar_matriz_aleatoria(h_B, n);

        // 1. SEQUENCIAL (CPU)
        printf("Calculando Sequencial (CPU)... "); fflush(stdout);
        clock_t inicio = clock();
        dgemm_seq(h_A, h_B, h_C_seq, n);
        clock_t fim = clock();

        double t_seq = ((double)(fim - inicio)) / CLOCKS_PER_SEC;
        double gflops_seq = (2.0 * n * n * n) / (t_seq * 1e9);
        printf("OK (%.4fs)\n", t_seq);
        fprintf(f_out, "%d,Sequencial,%.6f,%.2f,0.0\n", n, t_seq, gflops_seq);

        // PREPARAÇÃO GPU
        double *d_A, *d_B, *d_C;
        checkCudaError(cudaMalloc((void**)&d_A, bytes), "Malloc A");
        checkCudaError(cudaMalloc((void**)&d_B, bytes), "Malloc B");
        checkCudaError(cudaMalloc((void**)&d_C, bytes), "Malloc C");

        // Copia A e B para GPU
        checkCudaError(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "H2D A");
        checkCudaError(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "H2D B");

        dim3 dimBlock(TILE_SIZE, TILE_SIZE);
        dim3 dimGrid((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

        // 2. CUDA NAIVE
        checkCudaError(cudaMemset(d_C, 0, bytes), "Limpar C");
        cudaDeviceSynchronize();

        inicio = clock(); // Medindo com clock()
        dgemm_naive_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);
        checkCudaError(cudaGetLastError(), "Kernel Naive");
        cudaDeviceSynchronize(); // Espera GPU terminar para parar o relógio
        fim = clock();

        double t_naive = ((double)(fim - inicio)) / CLOCKS_PER_SEC;
        checkCudaError(cudaMemcpy(h_C_cuda, d_C, bytes, cudaMemcpyDeviceToHost), "D2H Naive");

        double delta_naive = calcular_delta(h_C_seq, h_C_cuda, n);
        double gflops_naive = (2.0 * n * n * n) / (t_naive * 1e9);
        printf("CUDA Naive: %.4fs | GFLOPS: %.2f | Delta: %e\n", t_naive, gflops_naive, delta_naive);
        fprintf(f_out, "%d,CUDA_Naive,%.6f,%.2f,%e\n", n, t_naive, gflops_naive, delta_naive);

        // 3. CUDA TILED
        checkCudaError(cudaMemset(d_C, 0, bytes), "Limpar C");
        cudaDeviceSynchronize();

        inicio = clock(); // Medindo com clock()
        dgemm_tiled_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);
        checkCudaError(cudaGetLastError(), "Kernel Tiled");
        cudaDeviceSynchronize();
        fim = clock();

        double t_tiled = ((double)(fim - inicio)) / CLOCKS_PER_SEC;
        checkCudaError(cudaMemcpy(h_C_cuda, d_C, bytes, cudaMemcpyDeviceToHost), "D2H Tiled");

        double delta_tiled = calcular_delta(h_C_seq, h_C_cuda, n);
        double gflops_tiled = (2.0 * n * n * n) / (t_tiled * 1e9);
        printf("CUDA Tiled: %.4fs | GFLOPS: %.2f | Delta: %e\n", t_tiled, gflops_tiled, delta_tiled);
        fprintf(f_out, "%d,CUDA_Tiled,%.6f,%.2f,%e\n", n, t_tiled, gflops_tiled, delta_tiled);

        // Limpeza
        free(h_A); free(h_B); free(h_C_seq); free(h_C_cuda);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    fclose(f_out);
    printf("\nBenchmark concluido.\n");
    return 0;
}
