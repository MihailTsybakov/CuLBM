/*
    LBM regular grid input-output functions
    - Read voxelized geometry 
    - Save output fields
*/

#ifndef LBMIO
#define LBMIO

#include <vector>
#include <string>
#include <fstream>
#include "misc.h"


std::vector<int> read_geometry_3D(std::string path, std::vector<int>& shape);

void write_data_3D(int Nx, int Ny, int Nz, const std::vector<std::string>& field_names, const std::vector<std::vector<float>>& values, std::string path);


#endif //LBMIO