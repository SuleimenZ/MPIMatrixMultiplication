#include <mpi.h>
#include <stdio.h>
#include <random>
#include <chrono>

void fillMatrix(int rows, int cols, float array[]) {
    int lowerBound = -100;  //min value for filling matrix
    int uppedBound = 100;   //max value for filling matrix

    std::uniform_real_distribution<float> uniformDistribution(-100, 100);
    std::default_random_engine re;
    //randomize seed
    re.seed(std::chrono::system_clock::now().time_since_epoch().count());

    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            array[row * cols + col] = uniformDistribution(re);
        }
    }
}

void displayMatrix(int rows, int cols, float result[]) {
    for (int row = 0; row < rows; row++) 
    {
        for (int col = 0; col < cols; col++) 
        {
            printf("%f ", result[row*cols + col]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char** argv) {
    int my_rank, size, root = 0;
    static const int row1 = 100;
    static const int col1 = 100;    //works as row2 too
    static const int col2 = 100;
    double startTime, finishTime;

    //I use one-dimension arrays, as I had problems instantiating two-dimension float arrays, but it shouldn't matter.
    //float* localResult = new float[row * line];
    float* matrixA = new float[row1 * col1];
    float* matrixB = new float[col1 * col2];
    float* matrixResult = new float[row1 * col2];

    MPI_Init(NULL, NULL);
    startTime = MPI_Wtime();
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int portion_size = (row1 * col1) / size;
    int localResult_size = row1 * col2 / size;
    float* localA = new float[portion_size];            //local array to gather data
    float* localResult = new float[localResult_size];   //local array to store result

    if (row1 % size != 0) {
        if (my_rank == 0) {
            printf("row1 should be divisable by size(number of processes) without remainder");
        }
        MPI_Finalize();
        return 0;
    }

    if (my_rank == 0) {
        //Finlling matricesx
        fillMatrix(row1, col1, matrixA);
        fillMatrix(col1, col2, matrixB);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    //Broadcasting matrixB to each rank
    MPI_Bcast(matrixB, col1 * col2, MPI_INT, root, MPI_COMM_WORLD);
    //Scattering portions of matrixA to each rank
    MPI_Scatter(matrixA, portion_size, MPI_INT, localA, portion_size, MPI_INT, root, MPI_COMM_WORLD);

    //Calculating local result
    //temp values are used so we don't have to recalculate same thing multiple times
    int tempMax = row1 * col2/size;
    for (int rowA = 0; rowA < tempMax; rowA++)
    {
        float sum = 0;
        int tempA = col1 * (rowA / col2);
        int tempB = rowA % col2;
        for (int rowB = 0; rowB < col1; rowB++)
        {
            sum += localA[tempA + rowB] * matrixB[rowB*col2 + tempB];
        }
        localResult[rowA] = sum;
    }

    //Gathering local results from each rank into matrixResult
    MPI_Gather(localResult, localResult_size, MPI_INT, matrixResult, localResult_size, MPI_INT, root, MPI_COMM_WORLD);
    finishTime = MPI_Wtime();
    MPI_Finalize();

    if (my_rank == 0) {
        //printf("///////////matrixA///////////\n");
        //displayMatrix(row1, col1, matrixA);
        //printf("///////////matrixB///////////\n");
        //displayMatrix(col1, col2, matrixB);
        //printf("///////////matrixResult///////////\n");
        //displayMatrix(row1, col2, matrixResult);
        printf("Elapsed time is %f ms\n", (finishTime - startTime)*1000); //time elasped in ms
    }
    return 0;
}