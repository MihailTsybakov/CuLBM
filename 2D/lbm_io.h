/*
 LBM Regular Grid Input-Output functions
    - Read the voxelized / pixelized model
    - Save data frame
*/

# ifndef LBMIO
# define LBMIO

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "misc.h"


std::vector<std::vector<int>> read_geometry_2D(std::string file_path);


void write_data_2D_vec(std::string save_path, int Nx, int Ny, std::vector<float> vx, std::vector<float> vy);


void write_data_2D_scal(std::string save_path, int Nx, int Ny, std::vector<float> v);


void write_data_2D_multi(int Nx, int Ny, const std::vector<std::string>& field_names, const std::vector<std::vector<float>>& field_values, const std::string& filename);

# endif //LBMIO