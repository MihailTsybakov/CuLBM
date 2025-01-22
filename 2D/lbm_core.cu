/*
    LBM Core functions
    (D2Q9 model)

    Coordinate system:

        (0,0)-----------> Y
          |
          |
          |
          |
          |
          V X

    Lattice weights:

    w = [1/36, 1/9, 1/36,
         1/9,  4/9, 1/9,
         1/36, 1/9, 1/36]

    Lattice Directions:
    
    e = [ (-1, -1), (-1, 0), (-1, 1),
          (0,  -1), (0,  0), (0,  1),
          (1,  -1), (1,  0), (1,  1) ]

*/

# include "lbm_core.h"


/*
        Lattice parameters
        
        Actually, they can be computed analytically in-place, but this will make
        the code less readable and understandable. Also, Defining weights and 
        directions on host seems to be more flexible for further modifications 
        to 3D case / other lattice models
*/
float W[9] = {
    1.0/36, 1.0/9, 1.0/36,
    1.0/9,  4.0/9, 1.0/9,
    1.0/36, 1.0/9, 1.0/36
};

int EX[9] = {
    -1, -1, -1, 
     0,  0,  0,
     1,  1,  1   
};

int EY[9] = {
    -1, 0, 1,
    -1, 0, 1,
    -1, 0, 1
};


std::vector<float> f_equi_host(int Nx, int Ny, std::vector<float> ux, std::vector<float> uy, std::vector<float> rho)
{   
    // Lattice sound speed
    float c = 1 / sqrt(3);

    // Computing velocity squared norm
    std::vector<float> u2(Nx*Ny, 0.0);
    for (int i = 0; i < Nx*Ny; i++)
    {
        u2[i] = (ux[i]*ux[i] + uy[i]*uy[i]) / (c*c);
    }

    // Computing velocity projections
    std::vector<float> u_proj(Nx*Ny*9, 0.0);
    for (int i = 0; i < 9; i++)
    {   
        float ex = EX[i], ey = EY[i];
        for (int ix = 0; ix < Nx; ix++)
        {
            for (int iy = 0; iy < Ny; iy++)
            {
                int lin_ind_3d = linear_index_3D(i, ix, iy, 9, Nx, Ny);
                int lin_ind    = linear_index(ix, iy, Nx, Ny);
                
                u_proj[lin_ind_3d] = (ex * ux[lin_ind] + ey * uy[lin_ind]) / (c*c); 
            } 
        }
    }

    // Computing equilibrium distribution
    std::vector<float> f_equi(Nx*Ny*9, 0.0);

    for (int i = 0; i < 9; i++)
    {
        for (int ix = 0; ix < Nx; ix++)
        {
            for (int iy = 0; iy < Ny; iy++)
            {
                int lin_ind_3d = linear_index_3D(i, ix, iy, 9, Nx, Ny);
                int lin_ind    = linear_index(ix, iy, Nx, Ny);

                f_equi[lin_ind_3d] = W[i]*rho[lin_ind]*(1 + u_proj[lin_ind_3d] + (u_proj[lin_ind_3d] * u_proj[lin_ind_3d]) / 2 - u2[lin_ind]/2);
            }
        }
    }

    return f_equi;
}





__global__ void streaming_device(float* d_f, float* d_f_buff, int* d_EX, int* d_EY, int Nx, int Ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix > 0  &&  iy > 0  &&  ix < Nx-1  &&  iy < Ny-1)
    {
        for (int i = 0; i < 9; i++)
        {
            int ex = d_EX[i];
            int ey = d_EY[i];

            int lin_ind_curr     = linear_index_3D(i, ix, iy, 9, Nx, Ny);
            int lin_ind_neighbor = linear_index_3D(i, ix - ex, iy - ey, 9, Nx, Ny);
            d_f_buff[lin_ind_curr] = d_f[lin_ind_neighbor];
        }
    }
}


__global__ void obstacles_device(float* d_f, float* d_obstacle, float* d_ux, float* d_uy, int Nx, int Ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= Nx || iy >= Ny) return;

    int linear_ind = linear_index(ix, iy, Nx, Ny);
    if (d_obstacle[linear_ind] > 0)
    {
        d_ux[linear_ind] = 0.0;
        d_uy[linear_ind] = 0.0;
        for (int i = 0; i < 9; i++)
        {
            int linear_ind_f    = linear_index_3D(i, ix, iy, 9, Nx, Ny);
            int linear_ind_op_f = linear_index_3D(8-i, ix, iy, 9, Nx, Ny); // !!!!!
            d_f[linear_ind_f] = d_f[linear_ind_op_f];
        }
    }
}


__global__ void recompute_u_rho_device(float* d_f, float* d_rho, float* d_ux, float* d_uy, int* d_EX, int* d_EY, int Nx, int Ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= Nx || iy >= Ny) return;

    float new_rho = 0.0;
    float new_ux  = 0.0;
    float new_uy  = 0.0;
    for (int i = 0; i < 9; i++)
    {

        int ex = d_EX[i];
        int ey = d_EY[i];

        int lin_ind = linear_index_3D(i, ix, iy, 9, Nx, Ny);
        new_rho    += d_f[lin_ind];
        new_ux     += d_f[lin_ind] * ex;
        new_uy     += d_f[lin_ind] * ey;
    }

    new_ux /= new_rho;
    new_uy /= new_rho;

    int lin_ind_2d    = linear_index(ix, iy, Nx, Ny);
    d_rho[lin_ind_2d] = new_rho;
    d_ux[lin_ind_2d]  = new_ux;
    d_uy[lin_ind_2d]  = new_uy;
}


__global__ void f_equi_device(float* d_f_buff, float* d_ux, float* d_uy, float* d_rho, float* d_W, int* d_EX, int* d_EY, int Nx, int Ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= Nx || iy >= Ny) return;
    
    int lin_ind_2d = linear_index(ix, iy, Nx, Ny);
    for (int i = 0; i < 9; i++)
    {
        float ex = d_EX[i];
        float ey = d_EY[i];
        int lin_ind_3d = linear_index_3D(i, ix, iy, 9, Nx, Ny);

        float u_sq   = (d_ux[lin_ind_2d]*d_ux[lin_ind_2d] + d_uy[lin_ind_2d]*d_uy[lin_ind_2d]) * 3;
        float u_proj = (ex * d_ux[lin_ind_2d] + ey * d_uy[lin_ind_2d]) * 3;

        d_f_buff[lin_ind_3d] = d_W[i] * d_rho[lin_ind_2d] * (1 + u_proj + (u_proj*u_proj)/2 - u_sq/2);
    }
}


__global__ void collision_device(float* d_f, float* d_f_buff, float tau_visc, int Nx, int Ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= Nx || iy >= Ny) return;

    for (int i = 0; i < 9; i++)
    {
        int lin_ind = linear_index_3D(i, ix, iy, 9, Nx, Ny);
        d_f[lin_ind] = d_f[lin_ind] + (d_f_buff[lin_ind] - d_f[lin_ind]) / (0.5 + tau_visc*3);
    }

}

__global__ void enforce_BC_device(float* d_f, const float* d_f_init, int Nx, int Ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if ((ix == 0 && iy < Ny)  || (iy == 0 && ix < Nx)  ||  (ix == Nx-1 & iy < Ny)  ||  (iy == Ny-1 && ix < Nx))
    {
        for (int i = 0; i < 9; i++)
        {
            int lin_ind = linear_index_3D(i, ix, iy, 9, Nx, Ny);
            d_f[lin_ind] = d_f_init[lin_ind];
        }
    }
}