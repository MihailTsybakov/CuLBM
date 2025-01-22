
# include "misc.h"


void send_log(std::string log_message)
{

    std::time_t now = std::time(nullptr);
    std::tm* local_time = std::localtime(&now);

    std::cout << " ["
              << std::setw(2) << std::setfill('0') << local_time->tm_mday << "-"
              << std::setw(2) << std::setfill('0') << (local_time->tm_mon + 1) << "-"
              << (local_time->tm_year + 1900) << ", "
              << std::setw(2) << std::setfill('0') << local_time->tm_hour << ":"
              << std::setw(2) << std::setfill('0') << local_time->tm_min << ":"
              << std::setw(2) << std::setfill('0') << local_time->tm_sec << "]  |  "
              << log_message
              << std::endl;
}