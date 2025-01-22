
# include "misc.h"


std::string time_stamp() 
{
    std::time_t now = std::time(nullptr);
    std::tm* localTime = std::localtime(&now);

    std::stringstream timestamp;
    timestamp << "[" 
              << std::setw(2) << std::setfill('0') << localTime->tm_mday << "-"
              << std::setw(2) << std::setfill('0') << (localTime->tm_mon + 1) << "-"
              << (localTime->tm_year + 1900) << " "
              << std::setw(2) << std::setfill('0') << localTime->tm_hour << ":"
              << std::setw(2) << std::setfill('0') << localTime->tm_min << ":"
              << std::setw(2) << std::setfill('0') << localTime->tm_sec
              << "]  |  ";

    return timestamp.str();
}

int estim_mem_usage(std::vector<int> grid_shape)
{
    int N = grid_shape[0] * grid_shape[1] * grid_shape[2];
    int grid_float = N * sizeof(float);

    int memory_usage = 0;

    memory_usage  = grid_float * 3;   // ux, uy, uz
    memory_usage += grid_float;       // rho
    memory_usage += grid_float;       // obstacle
    memory_usage += 2*15*grid_float;  // f & f_buffer

    memory_usage += 75*sizeof(float); // Lattice parameters :)

    return memory_usage; 
}