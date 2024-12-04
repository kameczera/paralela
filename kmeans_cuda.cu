#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>


#define NUM_POINTS 253681  // Ajuste para o número total de pontos na base de dados
#define NUM_DIMENSIONS 21   // Número de variáveis ou colunas numéricas
#define K 5
#define MAX_ITERATIONS 100
#define NUM_THREADS 16 //AQUI VOCÊ MUDA O NUMERO DE THREADS
#define NUM_TEAMS 16

__global__ void assign_labels_cuda(double* points, double* centroids, int* labels, int* changes, int num_points, int num_dimensions, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Memória compartilhada para mudanças locais do bloco
    __shared__ int shared_changes;

    // Inicializar a variável em memória compartilhada
    if (threadIdx.x == 0) {
        shared_changes = 0;
    }
    __syncthreads();

    if (idx < num_points) {
        int nearest_centroid = 0;
        double min_distance = 1e20;

        // Encontrar o centróide mais próximo
        for (int j = 0; j < k; j++) {
            double distance = 0.0;
            for (int d = 0; d < num_dimensions; d++) {
                double diff = points[idx * num_dimensions + d] - centroids[j * num_dimensions + d];
                distance += diff * diff;
            }
            if (distance < min_distance) {
                min_distance = distance;
                nearest_centroid = j;
            }
        }

        // Atualizar rótulo se necessário
        if (labels[idx] != nearest_centroid) {
            // Acumular mudanças na memória compartilhada
            labels[idx] = nearest_centroid;
            shared_changes++;
        }
    }    // Somar mudanças locais na variável global (feito apenas pelo thread 0 do bloco)
    if (threadIdx.x == 0) {
        *changes += shared_changes;
    }
}

void assign_labels_with_cuda(double points[NUM_POINTS][NUM_DIMENSIONS], int labels[NUM_POINTS], double centroids[K][NUM_DIMENSIONS], int* changes) {
    // Alocar memória no dispositivo
    double* d_points, * d_centroids;
    int* d_labels, * d_changes;

    cudaMalloc(&d_points, NUM_POINTS * NUM_DIMENSIONS * sizeof(double));
    cudaMalloc(&d_labels, NUM_POINTS * sizeof(int));
    cudaMalloc(&d_centroids, K * NUM_DIMENSIONS * sizeof(double));
    cudaMalloc(&d_changes, sizeof(int));

    // Copiar dados para o dispositivo
    cudaMemcpy(d_points, points, NUM_POINTS * NUM_DIMENSIONS * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels, NUM_POINTS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, K * NUM_DIMENSIONS * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_changes, changes, sizeof(int), cudaMemcpyHostToDevice);

    int threads_per_block = 128;
    int num_blocks = (NUM_POINTS + threads_per_block - 1) / threads_per_block;
    // Chamar o kernel CUDA
    double start = omp_get_wtime();
    assign_labels_cuda << <num_blocks, threads_per_block >> > (d_points, d_centroids, d_labels, d_changes, NUM_POINTS, NUM_DIMENSIONS, K);
    double end = omp_get_wtime();

    printf("Tempo de execução: %f segundos\n", end - start);

    cudaDeviceSynchronize();

    // Copiar resultados de volta para o host
    cudaMemcpy(labels, d_labels, NUM_POINTS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(changes, d_changes, sizeof(int), cudaMemcpyDeviceToHost);

    // Liberar memória do dispositivo
    cudaFree(d_points);
    cudaFree(d_labels);
    cudaFree(d_centroids);
    cudaFree(d_changes);
}

void kmeans_parallel(double points[NUM_POINTS][NUM_DIMENSIONS], int labels[NUM_POINTS], double centroids[K][NUM_DIMENSIONS]) {
    int iterations = 0;
    while (iterations < MAX_ITERATIONS) {
        int changes = 0;

        // Substituir a parte do OpenMP pelo CUDA
        assign_labels_with_cuda(points, labels, centroids, &changes);

        double new_centroids[K][NUM_DIMENSIONS] = { 0 };
        int counts[K] = { 0 };

        omp_set_num_threads(NUM_THREADS);
#pragma omp parallel
        {
            double local_centroids[K][NUM_DIMENSIONS] = { 0 };
            int local_counts[K] = { 0 };

#pragma omp for
            for (int i = 0; i < NUM_POINTS; i++) {
                int cluster = labels[i];
                local_counts[cluster]++;
                for (int d = 0; d < NUM_DIMENSIONS; d++) {
                    local_centroids[cluster][d] += points[i][d];
                }
            }

#pragma omp critical
            {
                for (int j = 0; j < K; j++) {
                    counts[j] += local_counts[j];
                    for (int d = 0; d < NUM_DIMENSIONS; d++) {
                        new_centroids[j][d] += local_centroids[j][d];
                    }
                }
            }
        }

        for (int j = 0; j < K; j++) {
            if (counts[j] > 0) {
                for (int d = 0; d < NUM_DIMENSIONS; d++) {
                    centroids[j][d] = new_centroids[j][d] / counts[j];
                }
            }
        }

        if (changes == 0) {
            break;
        }

        iterations++;
    }
}

int main() {
    double (*points)[NUM_DIMENSIONS] = (double (*)[NUM_DIMENSIONS])malloc(NUM_POINTS * sizeof(*points));
    int *labels = (int *)malloc(NUM_POINTS * sizeof(*labels));

    double centroids[K][NUM_DIMENSIONS];

    FILE *file = fopen("processed_data_diabetes.csv", "r");
    if (!file) {
        printf("Erro ao abrir o arquivo.\n");
        return 1;
    }

    for (int i = 0; i < NUM_POINTS; i++) {
        for (int j = 0; j < NUM_DIMENSIONS; j++) {
            fscanf(file, "%lf,", &points[i][j]);
        }
        labels[i] = 0;
    }
    fclose(file);

    for (int j = 0; j < K; j++) {
        for (int d = 0; d < NUM_DIMENSIONS; d++) {
            centroids[j][d] = rand() % 100; 
        }
    }

    double start = omp_get_wtime();
    kmeans_parallel(points, labels, centroids);
    double end = omp_get_wtime();

    printf("Tempo de execução: %f segundos\n", end - start);

    free(points);
    free(labels);

    return 0;
}