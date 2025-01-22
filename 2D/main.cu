/*
CUDA LBM 2D 
*/

# include "lbm_io.h"
# include "misc.h"
# include "lbm_core.h"


/*
    All arrays store data in 1D linear manner 
    Each linear 1D index is computed from nD with ascending order:
        arr.shape = [Nx, Ny]    --> i = ix + iy*Nx
        arr.shape = [N, Nx, Ny] --> i = j + ix*N + iy*N*Nx
*/


int main(void)
{
    send_log("CUDA LBM: Started");

    /*
        Reading Geometry file
    */
    std::string input_path = "C:/Data/Work/Code/LBM/LBM_2D_GPU/Data/test_geom_2.txt";
    std::string output_base = "C:/Data/Work/Code/LBM/LBM_2D_GPU/Data/Out/";
    auto geom_2d = read_geometry_2D(input_path); // vector<vector<int>>

    int Nx = geom_2d.size();
    int Ny = geom_2d[0].size();

    send_log("Grid cells X: ");
    std::cout << "   [" << std::to_string(Nx) << "]" << std::endl;

    send_log("Grid cells Y: ");
    std::cout << "   [" << std::to_string(Ny) << "]" << std::endl;



    /*
        Simulation parameters
    */
    float ux_0 = 0.10;
    float uy_0 = 0.00;

    float tau  = 0.01;

    int N_steps = 5000;
    int save_freq = 1000;
    bool dynamic_output = false;



    /*
        Arrays initialization
    */
    int N = Nx*Ny;
    std::vector<float> ux(N, ux_0);
    std::vector<float> uy(N, uy_0);
    std::vector<float> rho(N, 1.0);
    std::vector<float> f = f_equi_host(Nx, Ny, ux, uy, rho);
    std::vector<float> obstacle(N, 0.0);


    // Filling linear obstacle data 
    for (int ix = 0; ix < Nx; ix++)
    {
        for (int iy = 0; iy < Ny; iy++)
        {
            int i_1d = linear_index(ix, iy, Nx, Ny);
            obstacle[i_1d] = float(geom_2d[ix][iy]);
        }
    }



    /*
        Initializing GPU arrays & Allocating memory
    */
    float *d_ux, *d_uy, *d_rho, *d_f, *d_f_buff, *d_f_init, *d_obst;
    float *d_W;
    int   *d_EX, *d_EY;
    cudaMalloc(&d_ux,  N * sizeof(float));
    cudaMalloc(&d_uy,  N * sizeof(float));
    cudaMalloc(&d_rho, N * sizeof(float));
    cudaMalloc(&d_f,   9 * N * sizeof(float));
    cudaMalloc(&d_f_buff,   9 * N * sizeof(float));
    cudaMalloc(&d_f_init,   9 * N * sizeof(float));
    cudaMalloc(&d_obst, N * sizeof(float));

    cudaMalloc(&d_W, 9 * sizeof(float));
    cudaMalloc(&d_EX,9 * sizeof(int));
    cudaMalloc(&d_EY,9 * sizeof(int));
    
    send_log("Device memory allocated");



    /*
        Copying data to device
    */
    cudaMemcpy(d_ux, ux.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_uy, uy.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obst, obstacle.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho, rho.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f, f.data(), 9 * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f_init, f.data(), 9*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f_buff, f.data(), 9*N*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_W, W, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_EX, EX, 9 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_EY, EY, 9 * sizeof(int), cudaMemcpyHostToDevice);

    send_log("Initial data copied to device");


    /*
        Simulation loop
    */

    dim3 block_size(16, 16);
    dim3 grid_size((Nx + block_size.x - 1) / block_size.x, (Ny + block_size.y - 1) / block_size.y);

    send_log("Starting simulation...");
    
    send_log("Block size used: ");
    std::cout << "   [" << block_size.x << ", " << block_size.y << "]" << std::endl;

    send_log("Grid dimension: ");
    std::cout << "   [" << grid_size.x << ", " << grid_size.y << "]" << std::endl;

    send_log("Simulation steps: ");
    std::cout << "   [" << N_steps << "]" << std::endl;
    send_log("Output frequency: ");
    std::cout << "   [" << save_freq << "]" << std::endl;

    for (int i_step = 0; i_step < N_steps; i_step++)
    {  
        /*
            1. Streaming step 
        */
        streaming_device<<<grid_size, block_size>>>(d_f, d_f_buff, d_EX, d_EY, Nx, Ny);
        cudaDeviceSynchronize();
        float* temp = d_f;
        d_f = d_f_buff;
        d_f_buff = temp;
        
        /*
            2. Obstacle handling
        */
        obstacles_device<<<grid_size, block_size>>>(d_f, d_obst, d_ux, d_uy, Nx, Ny);

        /*  
            3. Density and velocity update
        */
        recompute_u_rho_device<<<grid_size, block_size>>>(d_f, d_rho, d_ux, d_uy, d_EX, d_EY, Nx, Ny);

        /*
            4. Collision step
        */
        f_equi_device<<<grid_size, block_size>>>(d_f_buff, d_ux, d_uy, d_rho, d_W, d_EX, d_EY, Nx, Ny);
        //cudaDeviceSynchronize();
        collision_device<<<grid_size, block_size>>>(d_f, d_f_buff, tau, Nx, Ny);

        /*
            5. Enforcing BC
        */
        enforce_BC_device<<<grid_size, block_size>>>(d_f, d_f_init, Nx, Ny);

        /*
            6. Output if needed
        */
        if (i_step % save_freq == 0 && dynamic_output == true)
        {
            std::cout << " Step [" << i_step << "] done" << std::endl;
            cudaMemcpy(ux.data(),  d_ux,  N*sizeof(float),   cudaMemcpyDeviceToHost);
            cudaMemcpy(uy.data(),  d_uy,  N*sizeof(float),   cudaMemcpyDeviceToHost);

            std::vector<std::string> field_names = {"vel_x", "vel_y"};
            std::vector<std::vector<float>> fields = {ux, uy};
            
            write_data_2D_multi(Nx, Ny, field_names, fields, output_base + "frame_" + std::to_string(i_step/save_freq) + ".txt");
        }
    }
    send_log("Simulation finished");

    /*
        Copying velocity field to host
    */
    cudaMemcpy(ux.data(), d_ux, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(uy.data(), d_uy, N*sizeof(float), cudaMemcpyDeviceToHost);
    send_log("Results copied to host");    


    /*
        Exporting to TXT
    */
    std::vector<std::string> field_names = {"vel_x", "vel_y"};
    std::vector<std::vector<float>> fields = {ux, uy};
        
    write_data_2D_multi(Nx, Ny, field_names, fields, output_base + "last_frame.txt");
    send_log("Results exported to txt");

    /*
        Freeing CUDA memory
    */
    cudaFree(d_ux);
    cudaFree(d_uy);
    cudaFree(d_rho);
    cudaFree(d_f);
    cudaFree(d_obst);

    cudaFree(d_W);
    cudaFree(d_EX);
    cudaFree(d_EY);

    send_log("Device memory free'd");

    return 0;
}