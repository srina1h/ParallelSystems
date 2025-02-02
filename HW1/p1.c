#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

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

    if (size != 2)
    {
        if (rank == 0)
        {
            fprintf(stderr, "This program requires exactly 2 processes.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    char *send_buffer = malloc(MAX_MSG_SIZE);
    char *recv_buffer = malloc(MAX_MSG_SIZE);
    memset(send_buffer, 'A', MAX_MSG_SIZE);

    for (int msg_size = MIN_MSG_SIZE; msg_size <= MAX_MSG_SIZE; msg_size *= 2)
    {
        double total_time = 0.0;

        for (int i = 0; i < NUM_ITERATIONS; i++)
        {
            struct timeval start_time, end_time;

            if (rank == 0)
            { // Sender
                gettimeofday(&start_time, NULL);

                MPI_Send(send_buffer, msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(recv_buffer, msg_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                gettimeofday(&end_time, NULL);
                total_time += get_elapsed_time(start_time, end_time);
            }
            else if (rank == 1)
            { // Receiver
                MPI_Recv(recv_buffer, msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(send_buffer, msg_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }

        if (rank == 0)
        {
            double avg_latency = total_time / NUM_ITERATIONS / 2;
            printf("Message Size: %d bytes | Avg Latency: %.3f us\n", msg_size, avg_latency);
        }
    }

    free(send_buffer);
    free(recv_buffer);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
