#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>

#define NUM_ITERATIONS 10
#define MIN_MSG_SIZE (32 * 1024)       // 32KB
#define MAX_MSG_SIZE (2 * 1024 * 1024) // 2MB

double get_elapsed_time(struct timeval start, struct timeval end)
{
    return (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int pair_rank;
    if (rank % 2 == 0)
    {
        pair_rank = rank + 1; // Even ranks communicate with the next odd rank
    }
    else
    {
        pair_rank = rank - 1; // Odd ranks communicate with the previous even rank
    }

    char *send_buffer = malloc(MAX_MSG_SIZE);
    char *recv_buffer = malloc(MAX_MSG_SIZE);
    memset(send_buffer, 'A', MAX_MSG_SIZE);

    for (int msg_size = MIN_MSG_SIZE; msg_size <= MAX_MSG_SIZE; msg_size *= 2)
    {
        double latencies[NUM_ITERATIONS];
        double total_time = 0.0;

        for (int i = 0; i < NUM_ITERATIONS; i++)
        {
            struct timeval start_time, end_time;

            if (rank % 2 == 0)
            { // Sender
                gettimeofday(&start_time, NULL);

                MPI_Send(send_buffer, msg_size, MPI_CHAR, pair_rank, 0, MPI_COMM_WORLD);
                MPI_Recv(recv_buffer, msg_size, MPI_CHAR, pair_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                gettimeofday(&end_time, NULL);
                latencies[i] = get_elapsed_time(start_time, end_time);
                total_time += latencies[i];
            }
            else
            { // Receiver
                MPI_Recv(recv_buffer, msg_size, MPI_CHAR, pair_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(send_buffer, msg_size, MPI_CHAR, pair_rank, 0, MPI_COMM_WORLD);
            }
        }

        if (rank % 2 == 0)
        { // Only print results from sender processes
            double avg_latency = total_time / NUM_ITERATIONS;

            double variance_sum = 0.0;
            for (int i = 0; i < NUM_ITERATIONS; i++)
            {
                variance_sum += pow(latencies[i] - avg_latency, 2);
            }
            double std_dev_latency = sqrt(variance_sum / NUM_ITERATIONS);

            printf("Rank %d <-> Rank %d | Message Size: %d bytes | Avg Latency: %.3f us | Std Dev: %.3f us\n",
                   rank, pair_rank, msg_size, avg_latency, std_dev_latency);
        }
    }

    if (send_buffer)
        free(send_buffer);
    if (recv_buffer)
        free(recv_buffer);

    MPI_Finalize();
    return EXIT_SUCCESS;
}