// Explicit diagnostic reset of the sole D435 camera; never touches robot control.
#include <librealsense2/rs.hpp>
#include <iostream>
#include <string>
int main() try {
    rs2::context context;
    auto devices=context.query_devices();
    if (devices.size()!=1) throw std::runtime_error("Expected exactly one RealSense camera");
    auto device=devices[0];
    if (std::string(device.get_info(RS2_CAMERA_INFO_NAME)).find("D435")==std::string::npos)
        throw std::runtime_error("Device is not a D435 camera");
    device.hardware_reset();
    std::cout << "D435 camera reset requested" << std::endl;
    return 0;
} catch(const std::exception& e) { std::cerr << e.what() << std::endl;return 1; }
