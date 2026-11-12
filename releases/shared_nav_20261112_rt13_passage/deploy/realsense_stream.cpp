// RGB-D acquisition only. No ROS or motor-control interfaces.
#include <librealsense2/rs.hpp>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

int main() try {
    // Perception consumes about 2 frames per second on the Jetson, so the full
    // 15 fps pair was pure USB traffic and CPU in align.process(). Six is the
    // lowest rate the D435i profile exposes and leaves three frames per slot.
    rs2::pipeline pipe;
    rs2::config config;
    config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 6);
    config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 6);
    auto profile = pipe.start(config);
    auto scale = profile.get_device().first<rs2::depth_sensor>().get_depth_scale();
    rs2::align align(RS2_STREAM_COLOR);
    auto in = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();
    std::cout << "{\"width\":" << in.width << ",\"height\":" << in.height
        << ",\"fx\":" << in.fx << ",\"fy\":" << in.fy << ",\"cx\":" << in.ppx
        << ",\"cy\":" << in.ppy << ",\"distortion_model\":" << int(in.model)
        << ",\"coefficients\":[";
    for (int i=0;i<5;i++) std::cout << (i ? "," : "") << in.coeffs[i];
    std::cout << "],\"fps\":6,\"depth_scale_m\":" << scale << "}" << std::endl;
    bool first = true;
    for (;;) {
        auto frames = align.process(pipe.wait_for_frames(first ? 15000 : 3000)).as<rs2::frameset>();
        first = false;
        auto color = frames.get_color_frame();
        auto depth = frames.get_depth_frame();
        double received = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        std::cout.write(reinterpret_cast<const char*>(&received), sizeof(received));
        for (int y=0;y<in.height;y++)
            std::cout.write(static_cast<const char*>(color.get_data())+y*color.get_stride_in_bytes(),in.width*3);
        const auto* raw = static_cast<const uint16_t*>(depth.get_data());
        std::vector<float> meters(in.width*in.height);
        for (int y=0;y<in.height;y++) for(int x=0;x<in.width;x++)
            meters[y*in.width+x]=raw[y*(depth.get_stride_in_bytes()/2)+x]*scale;
        std::cout.write(reinterpret_cast<const char*>(meters.data()),meters.size()*sizeof(float));
        std::cout.flush();
        if (!std::cout) break;
    }
    pipe.stop();
    return 0;
} catch (const std::exception& e) {
    std::cerr << "RGB-D capture failed: " << e.what() << std::endl;
    return 1;
}
