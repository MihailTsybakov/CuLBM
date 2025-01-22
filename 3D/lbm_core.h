/*
    3D LBM Core functions
    D3Q15 Model 

    Coordinate System:
    
         / Z
        /
       /
    (0,0,0)------> Y
      |
      |
      |
      V x 

    Lattice Directions:
    e = (
            [-1, -1, -1], [1, -1, -1], [0, 0, -1], [-1, 1, -1], [1, 1, -1],
            [-1, 0,   0], [0, -1,  0], [0, 0,  0], [0,  1,  0], [1, 0,  0],
            [-1, -1,  1], [1, -1,  1], [0, 0,  1], [-1, 1,  1], [1, 1,  1]
        )

    Lattice Weights:
    w = [
            1/72, 1/72, 1/9, 1/72, 1/72,
            1/9,  1/9,  2/9, 1/9,  1/9,
            1/72, 1/72, 1/9, 1/72, 1/72
        ]

    LBM Simulation step structure:
    
        1. Outflow Boundary Condition
        f[lattice_plane == (-1, y, z), x == x_end] = f[lattice_plane == (-1, y, z), x == x_end - dx]

        2. Density & Velocity update
        rho, ux, uy, uz = recompute_u_rho(f)

        3. Zou-He Inflow boundary condition
        ux[x == x_start] = ux_0
        rho[x == x_start] = ...

        4. Compute Equilibrium + Zou-He BC
        f_equi = f_equi_device(rho, ux, ...)
        fin[col1,0,:,:] = feq[col1,0,:,:] + fin[col3,0,:,:] - feq[col3,0,:,:]

        5. Collision step 
        f = f - (f - f_equi) / (1/2 + 3 * tau_visc)

        6. Bounce-Back boundary condition on obstacle
        f[obstacle == 1] = f_opposite[obstacle == 1]
        *
        * CAUTION: Possibly, collided state should be stored in d_f_buff, and non-collided state should be used for bounce back!
        * For now, will try without splitting
        * 

        7. Streaming step
        f[x,y,z] = f[x-dx, y-dy, z-dz]
*/


# ifndef LBM_CORE
# define LBM_CORE

# include <vector>
# include "misc.h"

/*
    Lattice parameters arrays
*/
extern float W[15];
extern int EX[15];
extern int EY[15];
extern int EZ[15];

extern int PLANE_1[5];
extern int PLANE_2[5];
extern int PLANE_3[5];


std::vector<float> f_equi_host(int Nx, int Ny, int Nz, 
                               const std::vector<float>& ux, const std::vector<float>& uy, const std::vector<float>& uz, 
                               const std::vector<float>& rho);


__global__ void outflow_device(float* d_f, int* d_PLANE_3, int Nx, int Ny, int Nz);

__global__ void update_u_rho_device(float* d_f, float* d_rho, 
                                    float* d_ux, float* d_uy, float* d_uz, 
                                    int* d_EX, int* d_EY, int* d_EZ, 
                                    int Nx, int Ny, int Nz);

__global__ void inflow_zou_he_device(float* d_f, float* d_rho, float* d_ux, int* d_PLANE_2, int* d_PLANE_3, float ux_0, int Nx, int Ny, int Nz);

__global__ void f_equi_device(float* d_f_buff, float* d_rho, 
                              float* d_ux, float* d_uy, float* d_uz, 
                              int* d_EX, int* d_EY, int* d_EZ, float* d_W,
                              int Nx, int Ny, int Nz);

__global__ void zou_he_equi_device(float* d_f, float* d_f_buff, int* d_PLANE_1, int* d_PLANE_3, int Nx, int Ny, int Nz);

__global__ void collision_device(float* d_f, float* d_f_buff, float tau_visc, int Nx, int Ny, int Nz);

__global__ void bounce_back_device(float* d_f, float* d_f_buff, int* d_obst, float* d_rho, float* d_ux, float* d_uy, float* d_uz, int Nx, int Ny, int Nz);

__global__ void streaming_device(float* d_f, float* d_f_buff, int* d_EX, int* d_EY, int* d_EZ, int Nx, int Ny, int Nz);

# endif//LBM_CORE