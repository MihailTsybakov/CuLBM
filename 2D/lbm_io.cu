/*
 Input - Output features for 
*/


# include "lbm_io.h"



std::vector<std::vector<int>> read_geometry_2D(std::string file_path)
{
    send_log("Reading 2D Geometry File:");
    std::cout << " [" << file_path << "]" << std::endl;

    std::ifstream file(file_path);
    if (!file.is_open())
    {
        send_log("Error: cannot open file");
    }

    int Nx, Ny;
    file >> Nx >> Ny;

    std::vector<std::vector<int>> geom(Nx, std::vector<int>(Ny, 0));

    int ix, iy, val;
    while (file >> ix >> iy >> val) {
        // Populate the grid
        geom[ix][iy] = val;
    }

    file.close();

    send_log("File scanned");
    return geom;
}



void write_data_2D_vec(std::string save_path, int Nx, int Ny, std::vector<float> vx, std::vector<float> vy)
{
    std::ofstream file(save_path);

    if (!file.is_open())
    {
        send_log("Error: cannot open file");
    }

    file << Nx << " " << Ny << "\n";

    for (int iy = 0; iy < Ny; ++iy) {
        for (int ix = 0; ix < Nx; ++ix) {
            int i = linear_index(ix, iy, Nx, Ny); 
            file << ix << " " << iy << " " << vx[i] << " " << vy[i] << "\n";
        }
    }

    file.close();
}



void write_data_2D_scal(std::string save_path, int Nx, int Ny, std::vector<float> v)
{
    std::ofstream file(save_path);

    if (!file.is_open())
    {
        send_log("Error: cannot open file");
    }

    file << Nx << " " << Ny << "\n";

    for (int iy = 0; iy < Ny; ++iy) {
        for (int ix = 0; ix < Nx; ++ix) {
            int i = linear_index(ix, iy, Nx, Ny); 
            file << ix << " " << iy << " " << v[i] << "\n";
        }
    }

    file.close();
}


void write_data_2D_multi(int Nx, int Ny, const std::vector<std::string>& field_names, const std::vector<std::vector<float>>& field_values, const std::string& filename) 
{
    // Check for consistency between fields
    int N_fields = field_names.size();
    if (field_values.size() != N_fields) {
        std::cerr << "Error: Number of field names and value arrays do not match!" << std::endl;
        return;
    }
    for (const auto& field : field_values) {
        if (field.size() != Nx * Ny) {
            std::cerr << "Error: Field size does not match grid dimensions!" << std::endl;
            return;
        }
    }

    // Open file
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing!" << std::endl;
        return;
    }

    // Write header: Nx, Ny, N_fields
    file << Nx << " " << Ny << " " << N_fields << "\n";

    // Write field names
    for (const auto& name : field_names) {
        file << name << " ";
    }
    file << "\n";

    // Write grid data
    for (int iy = 0; iy < Ny; ++iy) {
        for (int ix = 0; ix < Nx; ++ix) {
            int i = ix + iy * Nx; // Linear index
            file << ix << " " << iy; // Write grid indices
            for (int f = 0; f < N_fields; ++f) {
                file << " " << field_values[f][i]; // Write field values
            }
            file << "\n";
        }
    }

    file.close();
}