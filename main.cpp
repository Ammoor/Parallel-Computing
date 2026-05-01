#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>

using namespace std;

// ================= Utils =================

struct Config {
    string algo = "heat"; // heat | matmul
    bool nonBlocking = false;
};

Config parseArgs(int argc, char** argv) {
    Config config;
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--algo" && i + 1 < argc) {
            config.algo = argv[++i];
        } else if (arg == "--nonblock") {
            config.nonBlocking = true;
        }
    }
    return config;
}

vector<int> buildRowPartition(int totalRows, int size) {
    vector<int> rows(size, totalRows / size);
    int remainder = totalRows % size;
    for (int i = 0; i < remainder; i++) rows[i]++;
    return rows;
}

// ================= Heat Diffusion =================

void heatDiffusion(int rank, int size, bool nonBlocking) {
    int N = 1000;
    int iterations = 50;

    vector<int> rows = buildRowPartition(N, size);
    int localRows = rows[rank];

    vector<double> localGrid(localRows * N, 1.0);
    vector<double> newGrid = localGrid;

    MPI_Request requests[4];

    for (int it = 0; it < iterations; it++) {
        if (rank > 0) {
            if (nonBlocking)
                MPI_Isend(localGrid.data(), N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[0]);
            else
                MPI_Send(localGrid.data(), N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
        }

        if (rank < size - 1) {
            if (nonBlocking)
                MPI_Isend(&localGrid[(localRows - 1) * N], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[1]);
            else
                MPI_Send(&localGrid[(localRows - 1) * N], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
        }

        if (rank > 0) {
            if (nonBlocking)
                MPI_Irecv(localGrid.data(), N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[2]);
            else
                MPI_Recv(localGrid.data(), N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (rank < size - 1) {
            if (nonBlocking)
                MPI_Irecv(&localGrid[(localRows - 1) * N], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[3]);
            else
                MPI_Recv(&localGrid[(localRows - 1) * N], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (nonBlocking)
            MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

        for (int i = 1; i < localRows - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                newGrid[i * N + j] = 0.25 * (
                    localGrid[(i - 1) * N + j] +
                    localGrid[(i + 1) * N + j] +
                    localGrid[i * N + (j - 1)] +
                    localGrid[i * N + (j + 1)]
                );
            }
        }

        localGrid.swap(newGrid);
    }
}

// ================= Matrix Multiplication =================

void matrixMultiply(int rank, int size) {
    int N = 800;

    vector<int> rows = buildRowPartition(N, size);
    int localRows = rows[rank];

    vector<double> A(localRows * N, 1.0);
    vector<double> B(N * N, 1.0);
    vector<double> C(localRows * N, 0.0);

    // Transpose B for cache optimization
    vector<double> B_T(N * N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            B_T[j * N + i] = B[i * N + j];

    for (int i = 0; i < localRows; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B_T[j * N + k];
            }
            C[i * N + j] = sum;
        }
    }
}

// ================= Main =================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Config config = parseArgs(argc, argv);

    double start = MPI_Wtime();

    if (config.algo == "heat") {
        heatDiffusion(rank, size, config.nonBlocking);
    } else if (config.algo == "matmul") {
        matrixMultiply(rank, size);
    }

    double end = MPI_Wtime();

    double localTime = end - start;
    double maxTime;

    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Execution Time: " << maxTime << " seconds" << endl;
        cout << "Processes: " << size << endl;
        cout << "Algorithm: " << config.algo << endl;
        cout << "Mode: " << (config.nonBlocking ? "Non-Blocking" : "Blocking") << endl;
    }

    MPI_Finalize();
    return 0;
}
