#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <mpi.h> 
#include <assert.h>
#include <math.h>
#include <sys/time.h>

#define ROOT 0

typedef struct Cell {
	char state;	// state of cell: either D or A

	char up;	// states of neighboring cells
	char down;
	char left;
	char right;
	char upLeft;
	char upRight;
	char downLeft;
	char downRight;
} CELL;

void printCellStates(CELL cell);
void Simulate(int G, int p, int rank, int cols, int rows, CELL localRow[rows][cols], MPI_Status status);
CELL DetermineState(CELL cell);
void DisplayGoL(int p, int rank, int rows, int cols, CELL localRow[rows][cols], MPI_Status status);
int mod(int x, int y);
void badDisplay(int rank, int rows, int cols, CELL row[rows][cols], int mode);

struct timeval t7, t8;
int commTime = 0;

int main(int argc, char *argv[])
{
	int rank, p, i, j, localSeed;
	struct timeval t1, t2;
	MPI_Status status;

	gettimeofday(&t1,NULL);
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);	// get current rank
	MPI_Comm_size(MPI_COMM_WORLD, &p);	// get amount of processes

	srand(time(NULL));

	int globalSeeds[p];				

	assert(argc == 3);	

        // assign user variables
        int n = atoi(argv[1]);          // get matrix dimension
        int G = atoi(argv[2]);          // get number of generations

	if (rank == ROOT) {
		// populate seeds with random numbers
		for (i = 0; i < p; i++) {
			globalSeeds[i] = rand() % INT_MAX + 1;
		}
	}
	
	gettimeofday(&t7, NULL);

	// send each global seed to every other rank
	MPI_Scatter(&globalSeeds, 1, MPI_INT, &localSeed, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

	gettimeofday(&t8, NULL);
	commTime += (t8.tv_sec-t7.tv_sec)*1000 + (t8.tv_usec-t7.tv_usec)/1000;
	
	// locally create list of (n^2)/p random values
	int rows;
	int cols;
	if (n < p) { rows = 1; }
	else { rows = (int)(n / p); }
	
	if (pow(n, 2) < p) { cols = 1; }
	else { cols = (int)(pow(n, 2) / p) / rows; }
	
	int localMatrix[rows][cols];
	srand(localSeed);
	
	// assign each cell a random number
	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			localMatrix[i][j] = rand() % INT_MAX + 1;
		}
	}


	// determine status of all cells and store in localRow
	CELL localRow[rows][cols];
	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			if (localMatrix[i][j] % 2 == 0) {	// if number is even, cell is ALIVE
				localRow[i][j].state = 'A';
			}
			else {				// if number is odd, cell is DEAD
				localRow[i][j].state = 'D';
			}	
		}	
	}
	Simulate(G, p, rank, rows, cols, localRow, status);

	printf("\n\nTotal communication time: %d milliseconds\n\n", commTime);

	MPI_Finalize();
}

// runs generations and updates matrix
void Simulate(int G, int p, int rank, int rows, int cols, CELL localRow[rows][cols], MPI_Status status) {
	int i = 0, j = 0, k = 0, avgGenerationRuntime = 0, displayTime = 0;
	struct timeval t1, t2, t3, t4, t5, t6;
	
	// start total runtime
	gettimeofday(&t5, NULL);
	
	char top[255], bottom[255], row[255];
	char rTop[255], rBottom[255], rRow[255];

	// iterate generations
	i = 0;
	while (i < G) {
		// start timing current generation
		gettimeofday(&t1, NULL);

		// get neighbors
			
		for (i = 0; i < cols; i++) {
			top[i] = localRow[0][i].state;
			bottom[i] = localRow[rows - 1][i].state;
		}
     		
		gettimeofday(&t7, NULL);

		// send top row up
               	MPI_Send(top, cols + 1, MPI_CHAR, mod(rank - 1, p), 0, MPI_COMM_WORLD);
               	// send bottom row down
               	MPI_Send(bottom, cols + 1, MPI_CHAR, mod(rank + 1, p), 1, MPI_COMM_WORLD);
               	// recv bottom row from down
               	MPI_Recv(rBottom, cols + 1, MPI_CHAR, mod(rank + 1, p), 0, MPI_COMM_WORLD, &status);
               	// recv top row from up
               	MPI_Recv(rTop, cols + 1, MPI_CHAR, mod(rank - 1, p), 1, MPI_COMM_WORLD, &status);
     		
		gettimeofday(&t8, NULL);
        	commTime += (t8.tv_sec-t7.tv_sec)*1000 + (t8.tv_usec-t7.tv_usec)/1000;
                
		// assign top, bottom, left, and right neighbors
        	for (i = 0; i < rows; i++) {
                	for (j = 0; j < cols; j++) {
                                if (i == 0) {                   // if on first row of rank
                                        localRow[0][j].up = rTop[j];
                                        localRow[0][j].down = localRow[1][j].state;
	                        }
        	                else if (i == rows - 1) {       // if on last row of rank
              	                        localRow[rows - 1][j].down = rBottom[j];
                       	                localRow[rows - 1][j].up = localRow[rows - 2][j].state;
                               	}
                                else{                           // for any other row
    
    	                                localRow[i][j].up = localRow[i - 1][j].state;
               	                        localRow[i][j].down = localRow[i + 1][j].state;
                       	        }
                               	localRow[i][j].left = localRow[i][mod(j - 1, cols)].state;
                                localRow[i][j].right = localRow[i][mod(j + 1, cols)].state;
                       }
                }

	        // now that we have top and bottom neighbors
	        // assign diagonal neighbors
                for (i = 0; i < rows; i++) {
                	for (j = 0; j < cols; j++) {
                       	        localRow[i][j].upLeft = localRow[i][mod(j - 1, cols)].up;
                               	localRow[i][j].upRight = localRow[i][mod(j + 1, cols)].up;
	                        localRow[i][j].downLeft = localRow[i][mod(j - 1, cols)].down;
       	                        localRow[i][j].downRight = localRow[i][mod(j + 1, cols)].down;
               	        }
               	}


               	// iter thru all cells and determine new state
               	for (i = 0; i < rows; i++) {
                       	for (j = 0; j < cols; j++) {
                               	localRow[i][j] = DetermineState(localRow[i][j]);
                       	}
               	}
		
		// get end time of generation
		gettimeofday(&t2, NULL);	
		// get runtime of current generation
		avgGenerationRuntime += (t2.tv_sec-t1.tv_sec)*1000 + (t2.tv_usec-t1.tv_usec)/1000;
		
		// print matrix every other generation		
		if ((i % 2) == 0){ 
			// start display time
			gettimeofday(&t3, NULL);
			DisplayGoL(p, rank, rows, cols, localRow, status);
			gettimeofday(&t4, NULL);
			displayTime += (t4.tv_sec-t3.tv_sec)*1000 + (t4.tv_usec-t3.tv_usec)/1000;
		}
	
		printf("\n");
		i++;
	}

	printf("\n\nAvg generation runtime: %d milliseconds\n\n", avgGenerationRuntime / G);
	
	// end total runtime
	gettimeofday(&t6, NULL);
        int totalRuntime = ((t6.tv_sec-t5.tv_sec)*1000 + (t6.tv_usec-t5.tv_usec)/1000) - displayTime;
	printf("\n\nTotal runtime: %d milliseconds\n\n", totalRuntime);
	printf("\n\nTotal computation time: %d milliseconds\n\n", totalRuntime - commTime);
}


// display entire matrix
void DisplayGoL(int p, int rank, int rows, int cols, CELL localRow[rows][cols], MPI_Status status) {
	int i = 0, j = 0;
	
	// send each local matrix to root
	if (rank == 0) {
		// add local matrix to masterMatrix
		char masterMatrix[rows * p][cols];
		for (i = 0; i < rows; i++){
			for (j = 0; j < cols; j++){
				masterMatrix[i][j] = localRow[i][j].state;
			}
		}
		
		j = 1;
		// recv
		for (i = rows; i < rows * p; i += rows) {
			gettimeofday(&t7, NULL);

			MPI_Recv(&masterMatrix[i][0], rows * cols, MPI_CHAR, j++, 0,MPI_COMM_WORLD, &status);

        		gettimeofday(&t8, NULL);
        		commTime += (t8.tv_sec-t7.tv_sec)*1000 + (t8.tv_usec-t7.tv_usec)/1000;
		}
		
		// print master matrix
		for (i = 0; i < rows * p; i++){
			for (j = 0; j < cols; j++){
				printf("%c ", masterMatrix[i][j]);
			}
			printf("\n");
		}
	}
	else {
		char localMatrix[rows][cols];
		// make array of states
        	for (i = 0; i < rows; i++) {
                	for (j = 0; j < cols; j++) {
                        	localMatrix[i][j] = localRow[i][j].state;
                	}
        	}
		
		// send local matrix to root
		gettimeofday(&t7, NULL);
		
		MPI_Send(&localMatrix, rows * cols, MPI_CHAR, ROOT, 0, MPI_COMM_WORLD);

        	gettimeofday(&t8, NULL);
        	commTime += (t8.tv_sec-t7.tv_sec)*1000 + (t8.tv_usec-t7.tv_usec)/1000;
	}
}




// just prints a cell and its neighbors
void printCellStates(CELL cell) {
	printf("cell: %c, up: %c, down: %c, left: %c, right: %c, downLeft: %c, downRight: %c, upLeft: %c, upRight: %c\n\n", cell.up, cell.down, cell.left, cell.right, cell.downLeft, cell.downRight, cell.upLeft, cell.upRight);
}


// determines state of new cell
CELL DetermineState(CELL cell) {
	// count ALIVE neighbors
	int A_Count = 0;
	if (cell.up == 'A') A_Count++;
	if (cell.down == 'A') A_Count++;
	if (cell.left == 'A') A_Count++;
	if (cell.right == 'A') A_Count++;
	if (cell.upLeft == 'A') A_Count++;
	if (cell.upRight == 'A') A_Count++;
	if (cell.downLeft == 'A') A_Count++;
	if (cell.downRight == 'A') A_Count++;

	if (cell.state == 'A') {	// if cell is alive
		if (A_Count == 2 || A_Count == 3) {	// if cell has either 2 or 3 living cells, it lives on
			return cell;
		}	
		else {					// else, cell dies
			cell.state = 'D';
		}
	}
	else {				// if cell is dead
		if (A_Count >= 3) {		// if there are more than 3 living cells, cell lives
			cell.state = 'A';
		}
	}
	
	return cell;
}


int mod(int x, int y)
{
    int r = x % y;
    return r < 0 ? r + y : r;
}
