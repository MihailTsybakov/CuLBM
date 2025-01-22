/*
 Some helper functions
*/

#ifndef LBMMISC
#define LBMMISC

#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <vector>

std::string time_stamp();

int estim_mem_usage(std::vector<int> grid_shape);

__host__ __device__ inline int linear_index_3D(int ix, int iy, int iz, int NX, int NY, int NZ)
{
    return ix + iy*NX + iz*NX*NY;
}

__host__ __device__  inline int linear_index_4D(int ix, int iy, int iz, int ii, int Nx, int Ny, int Nz, int Ni)
{
    return ix + iy*Nx + iz*Nx*Ny + ii*Nx*Ny*Nz;
}


#endif //LBMMISC