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

    Lattice Directions:
    
    e = [ (-1, -1), (-1, 0), (-1, 1),
          (0,  -1), (0,  0), (0,  1),
          (1,  -1), (1,  0), (1,  1) ]


    Lattice weights:

    w = [1/36, 1/9, 1/36,
         1/9,  4/9, 1/9,
         1/36, 1/9, 1/36]


    LBM simulation step structure:
        
        1. Streaming step: flow distribution towards lattice directions:
            
            streaming_device<<<n_blocks, block_size>>>(...): 
                A) d_f --> d_f_buff                              (streams to the buffer)
            B) d_f = d_f_buff

        2. Obstacle handling step:

            obstacles_device<<<n_blocks, block_size>>>(...):
                A) d_ux[obstacle] = 0.0, d_uy[obstacle] = 0.0    (zeroes out velocities within the obstacle)
                B) d_f[i][obstacle] = d_f[i_opposite][obstacle]  (reverse-bounce the flow at boundaries)

        3. Updating density and velocity:
            
            recompute_u_rho_device<<<n_blocks, block_size>>>(...):
                A) d_rho = sum(d_f)                              (new density)
                B) d_ux  = sum(ex * d_f), d_uy = sum(ey * d_f)   (new velocity field)

        4. Collision step: Relaxing towards equilibrium

            4.1 f_equi_device<<<n_blocks, block_size>>>(...):
                A) d_ux, d_uy, d_rho --> d_f_equi                (equilibrium flow) - ACTUALLY d_f_buff WILL BE USED TO STORE d_f_equi !

            4.2 collision_device<<<n_blocks, block_size>>>(...):
                A) d_f = d_f + (d_f_equi - d_f) / (0.5 + tau_visc / c**2) (Relaxation)

        5. BC Step: 

            enforce_bc_device<<n_blocks, block_size>>>(...):
                A) d_f[boundary] = d_f_init[boundary]            (Enforcing inlet and outlet flows)

*/

# ifndef LBMCORE
# define LBMCORE 

# include <vector>
# include <cmath>

# include "misc.h"

// Lattice weights - contant, defined globally
extern float W[9];

// Lattice directions X - constant, defined globally
extern int EX[9];

// Lattice directions Y - constant, defined globally
extern int EY[9];



std::vector<float> f_equi_host(int Nx, int Ny, std::vector<float> ux, std::vector<float> uy, std::vector<float> rho);



__global__ void streaming_device(float* d_f, float* d_f_buff, int* d_EX, int* d_EY, int Nx, int Ny);

__global__ void obstacles_device(float* d_f, float* d_obstacle, float* d_ux, float* d_uy, int Nx, int Ny);

__global__ void recompute_u_rho_device(float* d_f, float* d_rho, float* d_ux, float* d_uy, int* d_EX, int* d_EY, int Nx, int Ny);

__global__ void f_equi_device(float* d_f_buff, float* d_ux, float* d_uy, float* d_rho, float* d_W, int* d_EX, int* d_EY, int Nx, int Ny); 

__global__ void collision_device(float* d_f, float* d_f_buff, float tau_visc, int Nx, int Ny);

__global__ void enforce_BC_device(float* d_f, const float* d_f_init, int Nx, int Ny);

# endif //LBMCORE