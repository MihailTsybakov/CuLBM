
# include "lbm_core.h"

/*
        Lattice parameters
        
        Actually, they can be computed analytically in-place, but this will make
        the code less readable and understandable. Also, Defining weights and 
        directions on host seems to be more flexible for further modifications 
        to other lattice models
*/
float W[15] = {
    1.0/72, 1.0/72, 1.0/9, 1.0/72, 1.0/72,
    1.0/9,  1.0/9,  2.0/9, 1.0/9,  1.0/9, 
    1.0/72, 1.0/72, 1.0/9, 1.0/72, 1.0/72
};

int EX[15] = {
    -1, 1, 0, -1, 1,
    -1, 0, 0,  0, 1,
    -1, 1, 0, -1, 1 
};

int EY[15] = {
    -1, -1, 0, 1, 1,
     0, -1, 0, 1, 0,
    -1, -1, 0, 1, 1
};

int EZ[15] = {
    -1, -1, -1, -1, -1,
    0,  0,  0,  0,  0,
    1,  1,  1,  1,  1
};

int PLANE_1[5] = {1, 4, 9, 11, 14}; // (1, y, z) plane 
int PLANE_2[5] = {2, 6, 7, 8,  12}; // (0, y, z) plane
int PLANE_3[5] = {0, 3, 5, 10, 13}; //(-1, y, z) plane

std::vector<float> f_equi_host(int Nx, int Ny, int Nz, 
                               const std::vector<float>& ux, const std::vector<float>& uy, const std::vector<float>& uz, 
                               const std::vector<float>& rho)
{
    float c = 1 / sqrt(3);

    std::vector<float> f_equi(Nx*Ny*Nz*15, 0.0);

    for (int ix = 0; ix < Nx; ix++)
    {
        for (int iy = 0; iy < Ny; iy++)
        {
            for (int iz = 0; iz < Nz; iz++)
            {
                int   grid_index = linear_index_3D(ix, iy, iz, Nx, Ny, Nz);
                float u2         = (ux[grid_index] * ux[grid_index] + uy[grid_index] * uy[grid_index] + uz[grid_index]*uz[grid_index]) / (c * c);

                for (int i = 0; i < 15; i++)
                {
                    int   lattice_index = linear_index_4D(ix, iy, iz, i, Nx, Ny, Nz, 15);
                    float u_proj        = (EX[i] * ux[grid_index] + EY[i] * uy[grid_index] + EZ[i] * uz[grid_index]) / (c * c);
                    
                    f_equi[lattice_index] = W[i] * rho[grid_index] * (1 + u_proj + u_proj*u_proj/2 - u2/2);
                }
            }
        }
    }

    return f_equi;
}


__global__ void outflow_device(float* d_f, int* d_PLANE_3, int Nx, int Ny, int Nz)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;

    if (ix != Nx-1 || iy >= Ny || iz >= Nz) return;

    for (int i = 0; i < 5; i++)
    {
        int lat_index_curr = linear_index_4D(ix,   iy, iz, d_PLANE_3[i], Nx, Ny, Nz, 15);
        int lat_index_nb   = linear_index_4D(ix-1, iy, iz, d_PLANE_3[i], Nx, Ny, Nz, 15); 
        d_f[lat_index_curr] = d_f[lat_index_nb];
    }
}


__global__ void update_u_rho_device(float* d_f, float* d_rho, 
                                    float* d_ux, float* d_uy, float* d_uz, 
                                    int* d_EX, int* d_EY, int* d_EZ, 
                                    int Nx, int Ny, int Nz)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) return;

    float rho_upd = 0.0;
    float ux_upd  = 0.0, uy_upd = 0.0, uz_upd = 0.0;

    int grid_index = linear_index_3D(ix, iy, iz, Nx, Ny, Nz);

    for (int i = 0; i < 15; i++)
    {
        int lattice_index = linear_index_4D(ix, iy, iz, i, Nx, Ny, Nz, 15);
        rho_upd += d_f[lattice_index];
        ux_upd  += d_EX[i] * d_f[lattice_index];
        uy_upd  += d_EY[i] * d_f[lattice_index];
        uz_upd  += d_EZ[i] * d_f[lattice_index];
    }

    ux_upd /= rho_upd;
    uy_upd /= rho_upd;
    uz_upd /= rho_upd;

    d_ux[grid_index] = ux_upd;
    d_uy[grid_index] = uy_upd;
    d_uz[grid_index] = uz_upd;
    d_rho[grid_index] = rho_upd;
}

__global__ void inflow_zou_he_device(float* d_f, float* d_rho, float* d_ux, int* d_PLANE_2, int* d_PLANE_3, float ux_0, int Nx, int Ny, int Nz)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;

    if (ix != 0 || iy >= Ny || iz >= Nz) return;

    int grid_index = linear_index_3D(ix,iy,iz, Nx, Ny, Nz);

    // Inflow velocity
    d_ux[grid_index] = ux_0;

    // Density correction
    float corr_term_1 = 0.0, corr_term_2 = 0.0;
    for (int i = 0; i < 5; i++)
    {
        int lattice_index_1 = linear_index_4D(ix, iy, iz, d_PLANE_2[i], Nx, Ny, Nz, 15);
        int lattice_index_2 = linear_index_4D(ix, iy, iz, d_PLANE_3[i], Nx, Ny, Nz, 15);

        corr_term_1 += d_f[lattice_index_1];
        corr_term_2 += d_f[lattice_index_2];
    }

    d_rho[grid_index] = 1 / (1 - ux_0) * (corr_term_1 + 2*corr_term_2);
}

__global__ void f_equi_device(float* d_f_buff, float* d_rho, 
                              float* d_ux, float* d_uy, float* d_uz, 
                              int* d_EX, int* d_EY, int* d_EZ, float* d_W,
                              int Nx, int Ny, int Nz)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) return;

    int   grid_index = linear_index_3D(ix, iy, iz, Nx, Ny, Nz);
    float u2         = (d_ux[grid_index] * d_ux[grid_index] + d_uy[grid_index] * d_uy[grid_index] + d_uz[grid_index] * d_uz[grid_index]) * 3;

    for (int i = 0; i < 15; i++)
    {
        int   lattice_index = linear_index_4D(ix, iy, iz, i, Nx, Ny, Nz, 15);
        float u_proj        = (d_EX[i] * d_ux[grid_index] + d_EY[i] * d_uy[grid_index] + d_EZ[i] * d_uz[grid_index]) * 3;

        d_f_buff[lattice_index] = d_W[i] * d_rho[grid_index] * (1 + u_proj + u_proj*u_proj/2 - u2/2);
    }
}

__global__ void zou_he_equi_device(float* d_f, float* d_f_buff, int* d_PLANE_1, int* d_PLANE_3, int Nx, int Ny, int Nz)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;

    if (ix != 0 || iy >= Ny || iz >= Nz) return;

    for (int i = 0; i < 5; i++)
    {
        int lattice_index_plane_1 = linear_index_4D(ix, iy, iz, d_PLANE_1[i], Nx, Ny, Nz, 15);
        int lattice_index_plane_3 = linear_index_4D(ix, iy, iz, d_PLANE_3[i], Nx, Ny, Nz, 15);

        d_f[lattice_index_plane_1] = d_f_buff[lattice_index_plane_1] + d_f[lattice_index_plane_3] - d_f_buff[lattice_index_plane_3];
    }
}

__global__ void collision_device(float* d_f, float* d_f_buff, float tau_visc, int Nx, int Ny, int Nz)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) return;

    for (int i = 0; i < 15; i++)
    {
        int lattice_index = linear_index_4D(ix, iy, iz, i, Nx, Ny, Nz, 15);
        d_f[lattice_index] = d_f[lattice_index] - (d_f[lattice_index] - d_f_buff[lattice_index]) / (0.5 + 3 * tau_visc);
    }
}

__global__ void bounce_back_device(float* d_f, float* d_f_buff, int* d_obst, float* d_rho, float* d_ux, float* d_uy, float* d_uz, int Nx, int Ny, int Nz)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) return;

    int grid_index = linear_index_3D(ix, iy, iz, Nx, Ny, Nz);

    if (d_obst[grid_index] != 0)
    {
        d_ux[grid_index] = 0.0;
        d_uy[grid_index] = 0.0;
        d_uz[grid_index] = 0.0;
        d_rho[grid_index] = 1.0;
        for (int i = 0; i < 15; i++)
        {
            int lattice_index          = linear_index_4D(ix, iy, iz, i, Nx, Ny, Nz, 15);
            int lattice_index_inverted = linear_index_4D(ix, iy, iz, 14-i, Nx, Ny, Nz, 15);
            
            d_f[lattice_index] = d_f[lattice_index_inverted];
        }
    }

}

__global__ void streaming_device(float* d_f, float* d_f_buff, int* d_EX, int* d_EY, int* d_EZ, int Nx, int Ny, int Nz)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;

    if (ix >= Nx-1 || iy >= Ny-1 || iz >= Nz-1 || ix == 0 || iy == 0 || iz == 0) return;

    for (int i = 0; i < 15; i++)
    {
        int dx = d_EX[i], dy = d_EY[i], dz = d_EZ[i];

        int lattice_index          = linear_index_4D(ix,   iy,    iz,     i, Nx, Ny, Nz, 15);
        int lattice_index_neighbor = linear_index_4D(ix-dx, iy-dy, iz-dz, i, Nx, Ny, Nz, 15);

        d_f_buff[lattice_index] = d_f[lattice_index_neighbor];
    }
}