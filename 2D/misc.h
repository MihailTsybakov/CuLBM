/*
 Some helper functions
*/

# ifndef LBMMISC
# define LBMMISC

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <ctime>

void send_log(std::string log_message);

__host__ __device__ inline int linear_index(int ix, int iy, int Nx, int Ny)
{
    return ix + iy*Nx;
}

__host__ __device__  inline int linear_index_3D(int ix, int iy, int iz, int Nx, int Ny, int Nz)
{
    return ix + iy*Nx + iz*Nx*Ny;
}

# endif //LBMMISC