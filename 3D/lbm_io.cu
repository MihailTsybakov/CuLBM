
#include "lbm_io.h"

std::vector<int> read_geometry_3D(std::string path, std::vector<int>& shape)
{
    std::cout << time_stamp() << "Reading geometry file: " << path << std::endl;

    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cout << time_stamp() << "Error: could not open file!" << std::endl;
    }

    int nx, ny, nz;
    file >> nx >> ny >> nz;
    shape[0] = nx; shape[1] = ny; shape[2] = nz;

    std::vector<int> geometry(nx*ny*nz, 0);

    int ix, iy, iz, val;
    while (file >> ix >> iy >> iz >> val)
    {
        int idx = linear_index_3D(ix, iy, iz, nx, ny, nz);
        geometry[idx] = val;
    }

    std::cout << time_stamp() << "File scanned. Grid shape: " << " [" << nx << ", " << ny << ", " << nz << "]" << std::endl;
    return geometry; 
}

void write_data_3D(int Nx, int Ny, int Nz, const std::vector<std::string>& field_names, const std::vector<std::vector<float>>& values, std::string path)
{
    std::ofstream file(path);
    if (!file.is_open())
    {
        std::cout << "Error: could not open file: [" << path << "]" << std::endl;
        return;
    }

    // Writing grid shape
    file << Nx << " " << Ny << " " << Nz << "\n";

    // Writing field names
    for (auto name : field_names)
    {
        file << name << " ";
    }
    file << "\n";

    // Writing data 
    for (int ix = 0; ix < Nx; ix++)
    {
        for (int iy = 0; iy < Ny; iy++)
        {
            for (int iz = 0; iz < Nz; iz++)
            {
                int idx = linear_index_3D(ix, iy, iz, Nx, Ny, Nz);
                file << ix << " " << iy << " " << iz << " ";
                for (int f = 0; f < values.size(); f++)
                {
                    file << values[f][idx] << " ";
                }
                file << "\n";
            }
        }
    }

    file.close();
}