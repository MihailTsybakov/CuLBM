/*
    CUDA LBM 3D
*/

# include "lbm_io.h"
# include "lbm_core.h"

int main(void)
{
    std::cout << time_stamp() << "CUDA LBM: Starting..." << std::endl;


    /*
        Reading Geometry
    */
    std::string model_path = "C:/Data/Work/Code/LBM/LBM_3D_GPU/Data/Input/droplet_1_TXT.txt";
    std::vector<int> grid_shape(3, 0);
    std::vector<int> geometry_vec = read_geometry_3D(model_path, grid_shape);
    int Nx = grid_shape[0], Ny = grid_shape[1], Nz = grid_shape[2];
    int N  = Nx*Ny*Nz;


    /*
        Simulation parameters

        The flow is considered to come in along X-axis. 
        Without loss of generality, any voxelized model can be oriented 
        according to x-axis flow
    */
    float reynolds = 15.0; 
    float flow_vel = 0.01; // Lattice units
    int char_size  = 50  ; // Characteristic size (for example, cylinder radius)
    float tau      = flow_vel * char_size / reynolds;
    std::cout << time_stamp() << "Computed viscosity: " << tau << std::endl;

    int N_steps         = 26;
    int save_freq       = 1000;
    bool dynamic_output = false;


    /* 
        Arrays initialization
    */
    std::vector<float> ux(N, flow_vel), uy(N, 0.0), uz(N, 0.0);
    std::vector<float> rho(N, 1.0);
    std::vector<float> f = f_equi_host(Nx, Ny, Nz, ux, uy, uz, rho);

    
    /*
        GPU Arrays initialization 
    */
    int mem_usage = estim_mem_usage(grid_shape);
    std::cout << time_stamp() << "Total grid dimension: " << N << " = " << float(N) / 1e6 << " M cells" << std::endl;
    std::cout << time_stamp() << "Device memory usage estimation: " << mem_usage << " bytes = " << float(mem_usage)/1048576.0 << " MB" << std::endl; // 2^20 = 1 MB

    float *d_ux, *d_uy, *d_uz, *d_rho, *d_f, *d_f_buff;
    float *d_W;
    int   *d_EX, *d_EY, *d_EZ, *d_PLANE_1, *d_PLANE_2, *d_PLANE_3, *d_obst;

    cudaMalloc(&d_ux, N * sizeof(float));
    cudaMalloc(&d_uy, N * sizeof(float));
    cudaMalloc(&d_uz, N * sizeof(float));
    cudaMalloc(&d_rho, N * sizeof(float));
    cudaMalloc(&d_f,  15*N*sizeof(float));
    cudaMalloc(&d_f_buff, 15*N*sizeof(float));
    cudaMalloc(&d_obst, N * sizeof(int));

    cudaMalloc(&d_W, 15*sizeof(float));
    cudaMalloc(&d_EX, 15*sizeof(int));
    cudaMalloc(&d_EY, 15*sizeof(int));
    cudaMalloc(&d_EZ, 15*sizeof(int));
    cudaMalloc(&d_PLANE_1, 5*sizeof(int));
    cudaMalloc(&d_PLANE_2, 5*sizeof(int));
    cudaMalloc(&d_PLANE_3, 5*sizeof(int));

    std::cout << time_stamp() << "Device memory allocated" << std::endl;


    /*
        Copying Data To Device
    */
    cudaMemcpy(d_ux, ux.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy, uy.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uz, uz.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho, rho.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f, f.data(), 15*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f_buff, f.data(), 15*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obst, geometry_vec.data(), N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_W, W, 15*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_EX, EX, 15*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_EY, EY, 15*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_EZ, EZ, 15*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_PLANE_1, PLANE_1, 5*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_PLANE_2, PLANE_2, 5*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_PLANE_3, PLANE_3, 5*sizeof(int), cudaMemcpyHostToDevice);

    std::cout << time_stamp() << "Data transferred to Device" << std::endl;


    /*
        Simulation Loop
    */
    std::cout << time_stamp() << "Starting simulation..." << std::endl;

    dim3 block_size(8, 8, 8);
    dim3 grid_size((Nx + block_size.x - 1) / block_size.x, (Ny + block_size.y - 1) / block_size.z, (Ny + block_size.z - 1) / block_size.z);
    
    std::cout << time_stamp() << "Block size: [" << block_size.x << ", " << block_size.y << ", " << block_size.z << "]" << std::endl;
    std::cout << time_stamp() << "Grid size:  [" << grid_size.x  << ", " << grid_size.y  << ", " << grid_size.z  << "]" << std::endl;
    std::cout << time_stamp() << "Simulation Steps: " << N_steps << std::endl;
    std::cout << time_stamp() << "Output frequency: " << save_freq << std::endl;

    for (int i = 0; i < N_steps; i++)
    {
        /*
            1. Outflow boundary condition
        */
        outflow_device<<<grid_size, block_size>>>(d_f, d_PLANE_3, Nx, Ny, Nz);
        cudaDeviceSynchronize();

        /*
            2. Recomputing U and Rho basing on flow distribution
        */
        update_u_rho_device<<<grid_size, block_size>>>(d_f, d_rho, d_ux, d_uy, d_uz, d_EX, d_EY, d_EZ, Nx, Ny, Nz);
        cudaDeviceSynchronize();

        /*
            3. Infow boundary condition
        */
        inflow_zou_he_device<<<grid_size, block_size>>>(d_f, d_rho, d_ux, d_PLANE_2, d_PLANE_3, flow_vel, Nx, Ny, Nz);
        cudaDeviceSynchronize();

        /*
            4.1 Computing equilibrium flow distribution
        */
        f_equi_device<<<grid_size, block_size>>>(d_f, d_rho, d_ux, d_uy, d_uz, d_EX, d_EY, d_EZ, d_W, Nx, Ny, Nz);
        cudaDeviceSynchronize();

        /*
            4.2 Applying Zou-He? boundary condition to flow distribution on inlet boundary
        */
        zou_he_equi_device<<<grid_size, block_size>>>(d_f, d_f_buff, d_PLANE_1, d_PLANE_3, Nx, Ny, Nz);
        cudaDeviceSynchronize();

        /*
            5. Collision step
        */
        collision_device<<<grid_size, block_size>>>(d_f, d_f_buff, tau, Nx, Ny, Nz);
        cudaDeviceSynchronize();

        /*
            6. Obstacle bounce-baclk boundary condition
        */
        bounce_back_device<<<grid_size, block_size>>>(d_f, d_f_buff, d_obst, d_rho, d_ux, d_uy, d_uz, Nx, Ny, Nz);
        cudaDeviceSynchronize();

        /*
            7. Streaming step
            Neighbor-dependent, so using dobule bufferization
        */
        streaming_device<<<grid_size, block_size>>>(d_f, d_f_buff, d_EX, d_EY, d_EZ, Nx, Ny, Nz);
        cudaDeviceSynchronize();
        float* temp = d_f;   d_f = d_f_buff;   d_f_buff = temp;

        /*
            9. Logging progress
        */
        if (i % save_freq == 0)
        {
            std::cout << time_stamp() << "Step " << i << " done." << std::endl;
        }
    }   


    /*
        Transferring last frame velocity to host
    */
    cudaMemcpy(ux.data(),  d_ux,  N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(uy.data(),  d_uy,  N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(uz.data(),  d_uz,  N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(rho.data(), d_rho, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<std::string>        field_names  = {"vel_x", "vel_y", "vel_z", "density"};
    std::vector<std::vector<float>> field_values = {ux, uy, uz, rho};
    std::string output_path                      = "C:/Data/Work/Code/LBM/LBM_3D_GPU/Data/Output/last_frame.txt";

    write_data_3D(Nx, Ny, Nz, field_names, field_values, output_path);
    std::cout << time_stamp() << "Data copied to host and exported to [" << output_path << "]" << std::endl;

    /*
        Device Memory Release
    */ 
    cudaFree(d_ux); 
    cudaFree(d_uy); 
    cudaFree(d_uz);
    cudaFree(d_rho);
    cudaFree(d_f);
    cudaFree(d_f_buff);
    cudaFree(d_obst);

    cudaFree(d_W);
    cudaFree(d_EX);
    cudaFree(d_EY);
    cudaFree(d_EZ);
    cudaFree(d_PLANE_1);
    cudaFree(d_PLANE_2);
    cudaFree(d_PLANE_3);

    std::cout << time_stamp() << "Device memory released" << std::endl;

    return 0;
}