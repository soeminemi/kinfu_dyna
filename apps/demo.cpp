#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/viz/vizcore.hpp>
#include <kfusion/kinfu.hpp>
#include <io/capture.hpp>
// #include "kfusion/marchingcubes.hpp"
// for measure
#include "cbodymodel.h"
#include "fitMesh.h"
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include "bodymeasurer.h"
#include <unistd.h>
#include <netdb.h>     //gethostbyname
#include <arpa/inet.h> //ntohl
#include <iostream>
#include <pthread.h>
#include <jsoncpp/json/json.h>
#include "CWebsocketServer.hpp"
#include <openssl/rsa.h>
#include <openssl/pem.h>
#include <openssl/err.h>
#include <ifaddrs.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <iomanip>
#include <sstream>
// #include <librealsense2/rs.hpp>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <string.h>
#include <thread>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <cmath>

using namespace kfusion;
#define COMBIN_MS // if body measurement is combined
bool flag_std_sample = false;
bool flag_show_image = false;
static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

static inline bool is_base64(unsigned char c)
{
    return (isalnum(c) || (c == '+') || (c == '/'));
}

unsigned char *base64_encode(const char *str0)
{
    unsigned char *str = (unsigned char *)str0; // 转为unsigned char无符号,移位操作时可以防止错误
    long len;                                   // base64处理后的字符串长度
    long str_len;                               // 源字符串长度
    long flag;                                  // 用于标识模3后的余数
    unsigned char *res;                         // 返回的字符串
    str_len = strlen((const char *)str);
    switch (str_len % 3) // 判断模3的余数
    {
    case 0:
        flag = 0;
        len = str_len / 3 * 4;
        break;
    case 1:
        flag = 1;
        len = (str_len / 3 + 1) * 4;
        break;
    case 2:
        flag = 2;
        len = (str_len / 3 + 1) * 4;
        break;
    }
    res = (unsigned char *)malloc(sizeof(unsigned char) * len + 1);
    for (int i = 0, j = 0; j < str_len - flag; j += 3, i += 4) // 先处理整除部分
    {
        // 注意&运算和位移运算的优先级,是先位移后与或非
        res[i] = base64_chars[str[j] >> 2];
        res[i + 1] = base64_chars[(str[j] & 0x3) << 4 | str[j + 1] >> 4];
        res[i + 2] = base64_chars[(str[j + 1] & 0xf) << 2 | (str[j + 2] >> 6)];
        res[i + 3] = base64_chars[str[j + 2] & 0x3f];
    }
    // 不满足被三整除时,要矫正
    switch (flag)
    {
    case 0:
        break; // 满足时直接退出
    case 1:
        res[len - 4] = base64_chars[str[str_len - 1] >> 2];         // 只剩一个字符时,右移两位得到高六位
        res[len - 3] = base64_chars[(str[str_len - 1] & 0x3) << 4]; // 获得低二位再右移四位,自动补0
        res[len - 2] = res[len - 1] = '=';
        break; // 最后两个补=
    case 2:
        res[len - 4] = base64_chars[str[str_len - 2] >> 2];                                 // 剩两个字符时,右移两位得高六位
        res[len - 3] = base64_chars[(str[str_len - 2] & 0x3) << 4 | str[str_len - 1] >> 4]; // 第一个字符低二位和第二个字符高四位
        res[len - 2] = base64_chars[(str[str_len - 1] & 0xf) << 2];                         // 第二个字符低四位,左移两位自动补0
        res[len - 1] = '=';                                                                 // 最后一个补=
        break;
    }
    res[len] = '\0'; // 补上字符串结束标识
    return res;
}

std::string base64_decode_str(const std::string &encoded_string)
{
    size_t in_len = encoded_string.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::string ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_]))
    {
        char_array_4[i++] = encoded_string[in_];
        in_++;
        if (i == 4)
        {
            for (i = 0; i < 4; i++)
                char_array_4[i] = base64_chars.find(char_array_4[i]) & 0xff;

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; (i < 3); i++)
                ret += char_array_3[i];
            i = 0;
        }
    }

    if (i)
    {
        for (j = 0; j < i; j++)
            char_array_4[j] = base64_chars.find(char_array_4[j]) & 0xff;

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);

        for (j = 0; (j < i - 1); j++)
            ret += char_array_3[j];
    }

    return ret;
}

// 获取本机MAC地址的函数
std::string get_local_mac()
{
    struct ifreq ifr;
    struct ifconf ifc;
    char buf[1024];
    int success = 0;

    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock == -1)
    {
        return "";
    }

    ifc.ifc_len = sizeof(buf);
    ifc.ifc_buf = buf;
    if (ioctl(sock, SIOCGIFCONF, &ifc) == -1)
    {
        close(sock);
        return "";
    }

    struct ifreq *it = ifc.ifc_req;
    const struct ifreq *const end = it + (ifc.ifc_len / sizeof(struct ifreq));

    for (; it != end; ++it)
    {
        strcpy(ifr.ifr_name, it->ifr_name);
        if (ioctl(sock, SIOCGIFFLAGS, &ifr) == 0)
        {
            if (!(ifr.ifr_flags & IFF_LOOPBACK))
            { // don't count loopback
                if (ioctl(sock, SIOCGIFHWADDR, &ifr) == 0)
                {
                    success = 1;
                    break;
                }
            }
        }
    }

    close(sock);

    if (success)
    {
        std::stringstream ss;
        ss << std::hex << std::setfill('0');
        for (int i = 0; i < 6; i++)
        {
            ss << std::setw(2) << static_cast<unsigned>(static_cast<unsigned char>(ifr.ifr_hwaddr.sa_data[i]));
            if (i != 5)
                ss << ":";
        }
        return ss.str();
    }

    return "";
}

class KinFuApp
{
public:
    int frame_idx = 0;
    string cloth_type = "tieshen";
    string measure_type = "qipao";
    string gender = "male";
    string device_type = "kinect";
    string spfile_folder = "./check_results/body_default/";
    string color_folder = "./check_results/body_default/colors";
    string depth_folder = "./check_results/body_default/depths";
    string align_color_folder = "./check_results/body_default/align_colors";
    KinFuParams params = KinFuParams::default_params();
    double fx;
    double fy;
    double cx;
    double cy;
    double k1 = 0,k2 = 0,k3 = 0;
    double p1 = 0, p2 = 0;
    int port = 9099;

    KinFuApp() : exit_(false), iteractive_mode_(false), pause_(true)
    {
        
        if (device_type == "kinect")
        {
            params.intr = Intr(365.3566f, 365.3566f, 261.4155f, 206.6168f,k1,k2,k3,p1,p2);
            params.cols = 512; // pixels
            params.rows = 414; // pixels
            fx = 365.3566f, fy = 365.3566f, cx = 261.4155f, cy = 206.6168f;
            p1 = 0;
            p2 = 0;

        }
        else if (device_type == "realsense")
        {
            params.cols = 1280; // pixels
            params.rows = 780;  // pixels
            params.intr = Intr(898.033f, 898.745f, 653.17f, 353.58f,k1,k2,k3,p1,p2);
            fx = 898.033f, fy = 898.745f, cx = 653.17f, cy = 353.58f;
            p1 = 0;
            p2 = 0;
        }
        kinfu_ = KinFu::Ptr(new KinFu(params));
        // capture_.setRegistration(true);
        // cv::viz::WCube cube(cv::Vec3d::all(0), cv::Vec3d(params.volume_size), true, cv::viz::Color::apricot());
        // viz.showWidget("cube", cube, params.volume_pose);
        // viz.showWidget("coor", cv::viz::WCoordinateSystem(0.1));
        // viz.registerKeyboardCallback(KeyboardCallback, this);
    }

    void show_depth(const cv::Mat &depth)
    {
        cv::Mat display;
        // cv::normalize(depth, display, 0, 255, cv::NORM_MINMAX, CV_8U);
        depth.convertTo(display, CV_8U, 255.0 / 65535);
        cv::imshow("Depth", display);
    }

    void show_raycasted(KinFu &kinfu)
    {
        const int mode = 0;
        if (iteractive_mode_)
            ; // kinfu.renderImage(view_device_, viz.getViewerPose(), mode);
        else
            kinfu.renderImage(view_device_, mode);

        view_host_.create(view_device_.rows(), view_device_.cols(), CV_8UC4);
        // points_host_.create(view_device_.rows(), view_device_.cols(), CV_32FC4);
        // kinfu.getPoints(points_host_);
        // std::stringstream ss;
        // ss<<"./results/rst"<<frame_idx<<".ply";
        // kinfu.toPly(points_host_,ss.str());
        view_device_.download(view_host_.ptr<void>(), view_host_.step);
        if (flag_show_image){
            cv::Mat rotated_view;
            cv::rotate(view_host_, rotated_view, cv::ROTATE_90_CLOCKWISE);
            cv::imshow("Scene", rotated_view);
        }
    }

    void take_cloud(KinFu &kinfu)
    {
        cuda::DeviceArray<Point> cloud = kinfu.tsdf().fetchCloud(cloud_buffer);
        cv::Mat cloud_host(1, (int)cloud.size(), CV_32FC4);
        cloud.download(cloud_host.ptr<Point>());
        // viz.showWidget("cloud", cv::viz::WCloud(cloud_host));
        // viz.showWidget("cloud", cv::viz::WPaintedCloud(cloud_host));
    }

    bool execute_ws()
    {
        string vcode = "none";
        bool flag_started = false;
        CWSServer ws;
        ws.set_port(port);
        thread t_ws(bind(&CWSServer::excute, &ws));
        bool flag_got = false;
        int fid = 0;
        KinFu &kinfu = *kinfu_;
        cv::Mat depth, image;
        double time_ms = 0;
        bool has_image = false;
        pause_ = false;
        Json::Value jv;
        Json::Reader jreader;
        static std::chrono::steady_clock::time_point lastValidMessageTime = std::chrono::steady_clock::now();
        while (true)
        {
            std::cout << "try to achieve msg" << std::endl;
            auto a = ws.archieve_message(flag_got);
            if (flag_got)
            {
                // start to process the message
                //  std::cout<<a.msg_str<<std::endl;
                auto msg = a.msg->get_payload();
                auto currentTime = std::chrono::steady_clock::now();
                auto elapsedTime = std::chrono::duration_cast<std::chrono::minutes>(currentTime - lastValidMessageTime);
                // 处理客户端没有正常结束的情况，没有收到finish消息
                if (vcode != "none" && elapsedTime > std::chrono::seconds(15))
                {
                    std::cout << "连接超时，重置为可用状态" << std::endl;
                    vcode = "none"; // 超时，重置vcode
                    lastValidMessageTime = std::chrono::steady_clock::now();
                    kinfu.reset();
                }
                // std::cout<<"msg received:"<<msg<<std::endl;
                jreader.parse(msg, jv);
                // 先处理ack消息
                if (jv["ack"].isString())
                {
                    struct timeval tv;
                    gettimeofday(&tv, NULL);
                    // save result and files to sample
                    stringstream ss;
                    ss << tv.tv_sec;
                    if (vcode == "none")
                        ws.send_msg(a.hdl, "OK");
                    else
                        ws.send_msg(a.hdl, ss.str());
                    continue;
                }
                // 记录获得有效消息的时间

                lastValidMessageTime = std::chrono::steady_clock::now();

                // 当消息为需要处理的实际消息时，判断验证码是否对应
                if (jv["vcode"].isString())
                {
                    jv["vcode"].asString();
                    if (vcode == "none")
                    {
                        vcode = jv["vcode"].asString();
                    }
                    if (vcode != jv["vcode"].asString())
                    {
                        ws.send_msg(a.hdl, "vcode not correct, refused");
                        continue;
                    }
                }
                else
                {
                    ws.send_msg(a.hdl, "no vcode contained, refused");
                    continue;
                }

                if (jv["cmd"].asString() == "finish")
                {
                    if (kinfu.isLoopClosed()==false)
                    {
                        // for test
                        Json::Value rt;
                        rt["error"] = "not loop closed";
                        Json::StreamWriterBuilder jswBuilder;
                        jswBuilder["emitUTF8"] = true;
                        std::unique_ptr<Json::StreamWriter> jsWriter(jswBuilder.newStreamWriter());

                        std::ostringstream os;
                        jsWriter->write(rt, &os);
                        ws.send_msg(a.hdl, os.str());
                        kinfu.reset();
                        continue;
                    }
                    kinfu.loopClosureOptimize();//执行闭环优化
                    std::cout << "msg:" << msg << std::endl;
                    vcode = "none"; // ready to receive new process
                    if (jv["gender"].isString())
                    {
                        if (jv["gender"].asString() == "male")
                        {
                            gender = "male";
                            meshFittor = &meshFittorMale;
                            std::cout << "set to male" << std::endl;
                        }
                        else
                        {
                            gender = "female";
                            meshFittor = &meshFittorFemale;
                            std::cout << "set to female" << std::endl;
                        }
                    }
                    else
                    {
                        std::cout << "failed to get gender" << std::endl;
                    }
                    if(jv["weight"].isDouble())
                    {
                        double weight = jv["weight"].asDouble();
                        meshFittor->setWeight(weight,true);
                        flag_weight_set = true;
                        weight_set = weight;
                    }
                    else{
                        std::cout << "failed to get weight" << std::endl;
                        meshFittor->setWeight(50.0,false);
                        flag_weight_set = false;
                    }
                    if (jv["measure_type"].isString())
                    {
                        measure_type = jv["measure_type"].asString();
                    }
                    if (jv["cloth_type"].isString())
                    {
                        cloth_type = jv["cloth_type"].asString();
                    }
                    flag_started = false;
                    std::cout << "finished and try to measure" << std::endl;
                    // save final cloud to file
                    cuda::DeviceArray<Point> cloud = kinfu.tsdf().fetchCloud(cloud_buffer);
                    if (cloud.size() > 0)
                    {
                        std::cout << "try to fetch normals" << std::endl;
                        kinfu.tsdf().fetchNormals(cloud, normal_buffer);
                        //
                        cv::Mat cloud_host(1, (int)cloud.size(), CV_32FC4);
                        cloud.download(cloud_host.ptr<Point>());
                        //
                        cv::Mat normal_host(1, (int)cloud.size(), CV_32FC4);
                        normal_buffer.download(normal_host.ptr<Point>());

#ifdef COMBIN_MS
                        // save to file for measurement
                        std::stringstream ss;
                        ss << pfile;
                        kinfu.toPlyColor(cloud_host, normal_host, spfile_folder+"origin_cloud.ply", 255, 0, 0);
                        // kinfu.toPlyColorFilter(cloud_host, normal_host, ss.str(), 255, 0, 0);
                        // string scmd = "cp ./results/origin_cloud.ply "+spfile_folder+"origin_cloud.ply";
                        // system(scmd.c_str());
                        // start measurement
                        auto rst = func(spfile_folder+"origin_cloud.ply");
                        ws.send_msg(a.hdl, rst);
#endif
                    }
                    else
                    {

                        // for test
                        Json::Value rt;
                        rt["error"] = "points not enough";
                        Json::StreamWriterBuilder jswBuilder;
                        jswBuilder["emitUTF8"] = true;
                        std::unique_ptr<Json::StreamWriter> jsWriter(jswBuilder.newStreamWriter());

                        std::ostringstream os;
                        jsWriter->write(rt, &os);
                        ws.send_msg(a.hdl, os.str());
                    }

                    kinfu.reset();
                }
                else
                {
                    if (flag_started == false)
                    {
                        std::cout << "first frame got" << std::endl;
                        // std::cout<<msg<<std::endl;
                        std::vector<std::string> keys;
                        for (const auto &key : jv.getMemberNames())
                        {
                            keys.push_back(key);
                        }

                        // 输出所有的key
                        for (const auto &key : keys)
                        {
                            std::cout << "key: " << key << std::endl;
                        }
                        struct timeval tv;
                        gettimeofday(&tv, NULL);
                        // save result and files to sample
                        stringstream ss;
                        // 测试数据存储到单独的文件夹
                        string name = std::to_string(int(tv.tv_sec));
                        //
                        // {'FocalLengthX': 367.04278564453125, 'FocalLengthY': 367.04278564453125, 'PrincipalPointX': 255.80419921875, 'PrincipalPointY': 203.5063018798828, 'RadialDistortionSecondOrder': 0.09293150156736374, 'RadialDistortionFourthOrder': -0.2737075090408325, 'RadialDistortionSixthOrder': 0.09219703823328018}
                        if(jv["intrinsics"].isObject())
                        {
                            auto iv = jv["intrinsics"];
                            if(iv["FocalLengthX"].isDouble())
                            {
                                fx = iv["FocalLengthX"].asDouble();
                            }
                            if(iv["FocalLengthY"].isDouble())
                            {
                                fy = iv["FocalLengthY"].asDouble();
                            }
                            if(iv["PrincipalPointX"].isDouble())
                            {
                                cx = iv["PrincipalPointX"].asDouble();
                            }
                            if(iv["PrincipalPointY"].isDouble())
                            {
                                cy = iv["PrincipalPointY"].asDouble();
                            }
                            if(iv["RadialDistortionSecondOrder"].isDouble())
                            {
                                k1 = iv["RadialDistortionSecondOrder"].asDouble();
                            }
                            if(iv["RadialDistortionFourthOrder"].isDouble())
                            {
                                k2 = iv["RadialDistortionFourthOrder"].asDouble();
                            }
                            if(iv["RadialDistortionSixthOrder"].isDouble())
                            {
                                k3 = iv["RadialDistortionSixthOrder"].asDouble();
                            }
                            params.intr.fx = fx;
                            params.intr.fy = fy;
                            params.intr.cx = cx;
                            params.intr.cy = cy;
                            // params.intr.k1 = k1;
                            // params.intr.k2 = k2;
                            // params.intr.k3 = k3;
                            // params.intr.p1 = p1;
                            // params.intr.p2 = p2;
                            std::cout << "set intr params with fx:" << fx << " fy:" << fy << " cx:" << cx << " cy:" << cy << " k1:" << k1 << " k2:" << k2 << " k3:" << k3 << std::endl;
                            kinfu_->set_params(params);
                        }
                        if (jv["name"].isString())
                        {
                            std::cout << "name:" << jv["name"].asString() << std::endl;
                            name = jv["name"].asString() + "_" + name;
                        }
                        if (jv["flag_test"].isBool() && jv["flag_test"].asBool() == true)
                        {
                            ss << "./check_results_test/body_" << name << "/";
                        }
                        else
                        {
                            // save result and files to sample
                            ss << "./check_results/body_" << name << "/";
                        }
                        std::cout << "save folder is : " << ss.str() << std::endl;
                        spfile_folder = ss.str();
                        string cmd_mkdir = "mkdir -p " + spfile_folder;
                        system(cmd_mkdir.c_str());
                        //save params to json file
                        if(jv["intrinsics"].isObject())
                        {
                            ofstream ofs(spfile_folder + "intrinsics.json");
                            ofs << jv["intrinsics"].toStyledString();
                            ofs.close();
                        }
                        color_folder = spfile_folder + "colors/";
                        depth_folder = spfile_folder + "depths/";
                        system(("mkdir -p " + color_folder).c_str());
                        system(("mkdir -p " + depth_folder).c_str());
                    }
                    flag_started = true;
                    if (jv["img_type"].asString() == "color")
                    {
                        std::string ws_str = base64_decode_str(jv["data"].asString());
                        std::vector<unsigned char> img_vec(ws_str.begin(), ws_str.end());
                        std::cout << "decode png" << std::endl;
                        cv::Mat color = cv::imdecode(img_vec, cv::IMREAD_COLOR);
                        cv::imwrite(color_folder + jv["frame_id"].asString() + ".jpg", color);
                    }
                    else
                    {
                        cout<<"process "<<jv["frame_id"].asString()<<endl;
                        std::string ws_str = base64_decode_str(jv["data"].asString());
                        std::cout << "base 64 decoded" << std::endl;
                        std::vector<unsigned char> img_vec(ws_str.begin(), ws_str.end());
                        std::cout << "decode png" << std::endl;
                        cv::Mat depth = cv::imdecode(img_vec, cv::IMREAD_ANYDEPTH);
                        cv::imwrite(depth_folder + jv["frame_id"].asString() + ".png", depth);
                        if(kinfu_->isLoopClosed() == false)
                        {
                            if (device_type == "realsense")
                            {
                                depth = depth / 4;
                            }
                            else if (device_type == "kinect")
                            {
                                depth = depth;
                            }
                            for (size_t i = 0; i < depth.rows; i++)
                            {
                                for (size_t j = 0; j < depth.cols; j++)
                                {
                                    if (depth.at<ushort>(i, j) > 2500)
                                    {
                                        depth.at<ushort>(i, j) = 0;
                                    }
                                    if(i<20 || i> 414-20)
                                    {
                                        depth.at<ushort>(i,j) = 0;
                                    }
                                }
                            }
                            ////////////////////////START//////////////////////////
                            // user specified code, for test to filter the point cloud
                            cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
                            cameraMatrix.at<double>(0, 0) = fx;
                            cameraMatrix.at<double>(1, 1) = fy;
                            cameraMatrix.at<double>(0, 2) = cx;
                            cameraMatrix.at<double>(1, 2) = cy;
                            cv::Mat distCoeffs = cv::Mat::zeros(1, 5, CV_64F);
                            distCoeffs.at<double>(0) = k1;
                            distCoeffs.at<double>(1) = k2;
                            distCoeffs.at<double>(2) = p1;
                            distCoeffs.at<double>(3) = p2;
                            distCoeffs.at<double>(4) = k3;
                            cv::Mat undistortedDepth;
                            std::cout << "try to undistort image" << std::endl;
                            cv::undistort(depth, undistortedDepth, cameraMatrix, distCoeffs);
                            std::cout << "undistored" << std::endl;
                            depth = undistortedDepth;
                            // cv::imwrite(depth_folder + jv["frame_id"].asString() + "_undistored.png", depth);
                            // cv::medianBlur(depth, depth, 5);
                            //------------
                            cv::Mat filteredDepth = cv::Mat::zeros(depth.rows, depth.cols, depth.type());
                            for (size_t i = 0; i < depth.rows; i++)
                            {
                                for (size_t j = 0; j < depth.cols; j++)
                                {
                                    ushort minDepth = USHRT_MAX, maxDepth = 0;
                                    for (int k = -1; k <= 1; k++)
                                    {
                                        for (int l = -1; l <= 1; l++)
                                        {
                                            int new_i = i + k;
                                            int new_j = j + l;
                                            if (new_i >= 0 && new_i < depth.rows && new_j >= 0 && new_j < depth.cols)
                                            {
                                                ushort depthValue = depth.at<ushort>(new_i, new_j);
                                                if (depthValue < minDepth) minDepth = depthValue;
                                                if (depthValue > maxDepth) maxDepth = depthValue;
                                            }
                                        }
                                    }
                                    if (maxDepth - minDepth > 30) // 如果深度最大值和最小值的差异过大
                                    {
                                        filteredDepth.at<ushort>(i, j) = 0;
                                    }
                                    else
                                    {
                                        filteredDepth.at<ushort>(i, j) = depth.at<ushort>(i, j);
                                    }
                                }
                            }
                            // 对深度图进行噪声过滤
                            // depth = filteredDepth;
                            // cv::Mat noiseFilteredDepth = cv::Mat::zeros(depth.rows, depth.cols, depth.type());
                            // const int windowSize = 5; // 使用5x5的窗口
                            // const int halfWindow = windowSize / 2;
                            // const float stdDevThreshold = 5.0f; // 标准差阈值
                            
                            // for(int i = halfWindow; i < depth.rows-halfWindow; i++) {
                            //     for(int j = halfWindow; j < depth.cols-halfWindow; j++) {
                            //         // 计算窗口内的均值和标准差
                            //         float sum = 0;
                            //         float sqSum = 0;
                            //         int count = 0;
                                    
                            //         for(int m = -halfWindow; m <= halfWindow; m++) {
                            //             for(int n = -halfWindow; n <= halfWindow; n++) {
                            //                 ushort val = depth.at<ushort>(i+m, j+n);
                            //                 if(val > 0) { // 只考虑有效深度值
                            //                     sum += val;
                            //                     sqSum += val * val;
                            //                     count++;
                            //                 }
                            //             }
                            //         }
                                    
                            //         if(count > 0) {
                            //             float mean = sum / count;
                            //             float variance = (sqSum / count) - (mean * mean);
                            //             float stdDev = sqrt(variance);
                                        
                            //             // 如果标准差大于阈值，认为是噪声区域
                            //             if(stdDev > stdDevThreshold) {
                            //                 noiseFilteredDepth.at<ushort>(i,j) = 0;
                            //             } else {
                            //                 noiseFilteredDepth.at<ushort>(i,j) = depth.at<ushort>(i,j);
                            //             }
                            //         }
                            //     }
                            // }
                            
                            // filteredDepth = noiseFilteredDepth;
                            depth = filteredDepth;
                            kinfu.append_depth_image(depth);
                            depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);
                            // depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);
                            {
                                SampledScopeTime fps(time_ms);
                                (void)fps;
                                has_image = kinfu(depth_device_);
                            }
                            if (flag_show_image)
                            {
                                if (has_image)
                                    show_raycasted(kinfu);
                                // cv::imshow("Image", depth);
                                int key = cv::waitKey(pause_ ? 0 : 3);
                            }
                        }
                    }
                }
            }
            else
            {
                // std::cout << "achieve msg failed" << std::endl;
                ;
            }
        }
        return true;
    }

    bool execute()
    {
        KinFu &kinfu = *kinfu_;
        cv::Mat depth, image;
        double time_ms = 0;
        bool has_image = false;

        std::vector<cv::String> depths; // store paths,
        std::vector<cv::String> images; // store paths,

        cv::glob("./data_kinfu/rotperson_1/depth", depths);
        cv::glob("./data_kinfu/rotperson_1/color", images);

        std::sort(depths.begin(), depths.end());
        std::sort(images.begin(), images.end());

        pause_ = true;
        for (int i = 200; i < depths.size() && !exit_; ++i)
        {
            // if(i>200)
            //     exit_ = true;
            frame_idx = i;
            std::cout << "frame: " << i << std::endl;
            // bool has_frame = capture_.grab(depth, image);
            image = cv::imread(images[i], cv::IMREAD_COLOR);
            depth = cv::imread(depths[i], cv::IMREAD_ANYDEPTH);
            depth = depth / 4;
            ////////////////////////START//////////////////////////
            // user specified code, for test to filter the point cloud
            for (size_t i = 0; i < depth.rows; i++)
            {
                for (size_t j = 0; j < depth.cols; j++)
                {
                    if (depth.at<ushort>(i, j) > 2000)
                    {
                        depth.at<ushort>(i, j) = 0;
                    }
                }
            }
            // cv::Rect maskroi(0,0,200,720);
            // depth(maskroi) = 0;
            ////////////////////////END//////////////////////////
            depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);
            // depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);
            {
                SampledScopeTime fps(time_ms);
                (void)fps;
                has_image = kinfu(depth_device_);
            }

            if (has_image)
                show_raycasted(kinfu);

            // show_depth(depth);
            cv::imshow("Image", image);

            // if (!iteractive_mode_)
            //     viz.setViewerPose(kinfu.getCameraPose());

            int key = cv::waitKey(pause_ ? 0 : 3);

            switch (key)
            {
            case 't':
            case 'T':
                take_cloud(kinfu);
                break;
            case 'i':
            case 'I':
                iteractive_mode_ = !iteractive_mode_;
                break;
            case 's':
                pause_ = false;
                break;
            case 27:
                exit_ = true;
                break;
            case 'p':
                pause_ = !pause_;
                break;
            }

            // exit_ = exit_ || i > 100;
            //  viz.spinOnce(3, true);
            std::cout << "finish frame" << std::endl;
        }
        // save final cloud to file
        cuda::DeviceArray<Point> cloud = kinfu.tsdf().fetchCloud(cloud_buffer);
        kinfu.tsdf().fetchNormals(cloud, normal_buffer);
        //
        cv::Mat cloud_host(1, (int)cloud.size(), CV_32FC4);
        cloud.download(cloud_host.ptr<Point>());
        //
        cv::Mat normal_host(1, (int)cloud.size(), CV_32FC4);
        normal_buffer.download(normal_host.ptr<Point>());

#ifdef COMBIN_MS
        // save to file for measurement
        std::stringstream ss;
        ss << pfile;
        kinfu.toPlyColorFilter(cloud_host, normal_host, ss.str(), 255, 0, 0);
        // start measurement

        func(pfile);
#endif

        return true;
    }

    bool pause_ /*= false*/;
    bool exit_, iteractive_mode_;
    // OpenNISource& capture_;
    KinFu::Ptr kinfu_;
    // cv::viz::Viz3d viz;

    cv::Mat view_host_;
    cv::Mat points_host_;
    cuda::Image view_device_;
    cuda::Depth depth_device_;
    cuda::DeviceArray<Point> cloud_buffer;
    cuda::DeviceArray<Normal> normal_buffer;

#ifdef COMBIN_MS
    BodyMeasurer bm;
    fitMesh *meshFittor;
    fitMesh meshFittorMale;
    fitMesh meshFittorFemale;
    string pfile;
    // pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_filtered;
    string readFileIntoString(char *filename)
    {
        ifstream ifile(filename);
        // 将文件读入到ostringstream对象buf中
        ostringstream buf;
        char ch;
        while (buf && ifile.get(ch))
            buf.put(ch);
        // 返回与流对象buf关联的字符串
        return buf.str();
    }

    static string getCurrentTimeStr()
    {
        time_t t = time(NULL);
        char ch[64] = {0};
        strftime(ch, sizeof(ch) - 1, "%Y-%m-%d %H:%M:%S", localtime(&t)); // 年-月-日 时-分-秒
        return ch;
    }
    string func(string bodypath)
    {
        auto start = std::chrono::high_resolution_clock::now();
        ofstream ff("log.txt", ios::app);
        auto start_time = getCurrentTimeStr();
        ff << "calling the service @ " << start_time<< endl;
        ff.close();
        if (flag_std_sample)
        {
            cloth_type = "nake";
        }
        Json::Reader reader;
        Json::Value root;
        Json::Value rt;
        {
            if (gender == "male")
            {
                meshFittor = &meshFittorMale;
            }
            else
            {
                meshFittor = &meshFittorFemale;
            }
        }

        // cloth_type = "lvekuansong";

        cout << "step 1. load ply file:" << bodypath << endl;
        // step 1. load ply file
        Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Zero();
        if (flag_std_sample == true)
        {
            transformation_matrix(0, 0) = 1;
            transformation_matrix(1, 1) = 1;
            transformation_matrix(2, 2) = 1;
            transformation_matrix(3, 3) = 1;
        }
        else
        {
            transformation_matrix(0, 1) = -1;
            transformation_matrix(1, 0) = -1;
            transformation_matrix(2, 2) = -1;
            transformation_matrix(3, 3) = 1;
        }
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_orig(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        if (pcl::io::loadPLYFile<pcl::PointXYZRGBNormal>(bodypath, *cloud_orig) == -1)
        {
            return "{\"error\":\"can not load point cloud\"}";
        }

        if (cloud_orig->size() < 10000)
        {
            return "{\"error\":\"points not enough\"}";
        }
        std::cout << "ply file loaded, try to filter the data" << std::endl;
        pcl::transformPointCloudWithNormals(*cloud_orig, *cloud, transformation_matrix);

        // 创建用于平面检测的点云
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr detection_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        for (const auto& point : cloud->points) {
            if (point.y < -0.5) {
                detection_cloud->points.push_back(point);
            }
        }
        // 保存detection_cloud为ply到results文件夹中
        std::string detection_cloud_path = "results/detection_cloud.ply";
        pcl::io::savePLYFile(detection_cloud_path, *detection_cloud);
        std::cout << "Detection cloud saved to: " << detection_cloud_path << std::endl;

        // 使用RANSAC方法找到地面平面
        pcl::SACSegmentation<pcl::PointXYZRGBNormal> seg;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        
        // 配置RANSAC参数
        seg.setInputCloud(detection_cloud);
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);  // 设置模型为垂直平面
        seg.setMethodType(pcl::SAC_RANSAC);     // 使用RANSAC方法
        seg.setDistanceThreshold(0.1);         // 设置内点阈值为10cm
        
        // 设置平面法向约束,使其接近y轴方向
        Eigen::Vector3f axis = Eigen::Vector3f(0.0, 1.0, 0.0);
        seg.setAxis(axis);
        seg.setEpsAngle(20.0f * (M_PI/180.0f)); // 允许20度的偏差
        
        seg.segment(*inliers, *coefficients);

        // 检查是否找到地面平面
        if (inliers->indices.size() == 0)
        {
            std::cout << "No ground plane found." << std::endl;
            return "{\"error\":\"can not find ground plane\"}";
        }
        else
        {
            // 创建新的点云来存储过滤后的结果
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
            
            // 获取平面方程系数 ax + by + cz + d = 0
            float a = coefficients->values[0];
            float b = coefficients->values[1];
            float c = coefficients->values[2];
            float d = coefficients->values[3];
            
            // 遍历所有点,保留距离地面6cm以上的点
            for (size_t i = 0; i < cloud->points.size(); ++i)
            {
                const auto& point = cloud->points[i];
                // 计算点到平面的距离,带符号
                float distance = (a * point.x + b * point.y + c * point.z + d) / 
                            std::sqrt(a * a + b * b + c * c);
                
                // 只保留平面上方的点
                if (distance < -0.04)
                {
                    filtered_cloud->points.push_back(point); // 保留原始点的颜色和法向
                }
            }
            
            filtered_cloud->width = filtered_cloud->points.size();
            filtered_cloud->height = 1;
            filtered_cloud->is_dense = false;
            
            // 更新点云
            cloud = filtered_cloud;
            // 找到最小y值
            float min_y = std::numeric_limits<float>::max();
            float sum_x = 0, sum_z = 0;
            int point_count = 0;
            
            for(const auto& point : filtered_cloud->points) {
                if(point.y < min_y) {
                    min_y = point.y;
                }
            }
            for(const auto& point : filtered_cloud->points) {
                if(point.y > min_y + 0.1) { // 只统计y值在min_y+10cm以上的点
                    sum_x += point.x;
                    sum_z += point.z;
                    point_count++;
                }
            }
            // 计算中心点坐标
            float center_x = sum_x / point_count;
            float center_z = sum_z / point_count;
            
            // 创建新的点云来存储过滤后的结果
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr circle_filtered(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
            
            // 过滤掉y值小于min_y+10cm的点和距离中心30cm以外的点
            for(const auto& point : filtered_cloud->points) {
                { // y值需要大于min_y+10cm
                    float dx = point.x - center_x;
                    float dz = point.z - center_z;
                    float distance = std::sqrt(dx*dx + dz*dz);
                    
                    if(distance <= 0.3 || point.y > min_y + 0.1) { // 距离中心30cm以内的点
                        circle_filtered->points.push_back(point);
                    }
                }
            }
            
            circle_filtered->width = circle_filtered->points.size();
            circle_filtered->height = 1;
            circle_filtered->is_dense = false;
            
            // 更新点云
            cloud = circle_filtered;
            // 保存去除地平面后的点云
            pcl::io::savePLYFile("./results/no_ground.ply", *cloud);
        }

        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        // for(size_t i = 0;i<cloud->size();i++)
        // {
        //     pcl::PointXYZRGBNormal &tp = (*cloud)[i];
        //     tp.normal_x = -tp.normal_x;
        //     tp.normal_y = -tp.normal_y;
        //     tp.normal_z = -tp.normal_z;
        // }

        
        cout<<"step 2 filter the outlier points"<<endl;
        pcl::RadiusOutlierRemoval<pcl::PointXYZRGBNormal> sor;
        sor.setInputCloud(cloud);
        sor.setRadiusSearch(0.08);  // 增大搜索半径到8cm
        sor.setMinNeighborsInRadius(20);  // 增加最小邻居点数到20个
        sor.setNegative(false);
        sor.filter(*cloud_filtered);
        
        // 进行第二次过滤以进一步去除离群点
        pcl::RadiusOutlierRemoval<pcl::PointXYZRGBNormal> sor2;
        sor2.setInputCloud(cloud_filtered);
        sor2.setRadiusSearch(0.05);  // 使用较小的搜索半径进行精细过滤
        sor2.setMinNeighborsInRadius(15);  // 设置适中的邻居点数阈值
        sor2.setNegative(false);
        sor2.filter(*cloud_filtered);
        
        pcl::io::savePLYFile("./results/aft_filter.ply",*cloud_filtered);

        pcl::PointCloud<pcl::PointXYZRGBNormal> scan;
        scan = *cloud_filtered;
        // step 3. fit the model
        //  scan = *cloud;
        double maxz = -10000000, minz = 10000000;
        for (int i = 0; i < scan.points.size(); i++)
        {
            double z = scan.points[i].y;
            if (z > maxz)
                maxz = z;
            if (z < minz)
                minz = z;
        }
        pcl::io::savePLYFile("./results/scan.ply", scan);
        Json::Value val_weights;
        ifstream rcf("./data/weights_" + cloth_type + ".conf");
        if (!reader.parse(rcf, val_weights))
        {
            cout << "===================================================load config file for measurment failed" << endl;
        }
        else
        {
            cout << "config file is: " << cloth_type << endl;
            meshFittor->setWeights(val_weights["weight_out"].asDouble(), val_weights["weight_in"].asDouble(), val_weights["weight_in_margin"].asDouble(), val_weights["in_thick"].asDouble());
            cout << "===============================================weights seted" << endl;
        }
        cout << "reset parameters" << endl;
        meshFittor->resetParameters();
        cout << "start mainprocess" << endl;
        auto start_mp = std::chrono::high_resolution_clock::now();
        meshFittor->mainProcess(scan);
        auto end_mp  = std::chrono::high_resolution_clock::now();
        auto duration_mp = std::chrono::duration_cast<std::chrono::milliseconds>(end_mp - start_mp);
        std::cout << "主处理过程耗时: " << duration_mp.count() << " 毫秒" << std::endl;
        cout << "start measure" << endl;
        auto rst = measure_body(minz, maxz);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        cout << "time used for optimization: " << elapsed.count() << endl;
        cout <<"calling the service @ " << start_time<< endl;
        cout <<"end calling service @ "<<getCurrentTimeStr()<<endl;
        return rst;
    }
    std::string readFileToString(std::string filePath)
    {
        std::ifstream fileStream(filePath);
        if (!fileStream.is_open())
        {
            throw std::runtime_error("Unable to open file.");
        }
        std::stringstream stringStream;
        stringStream << fileStream.rdbuf();
        return stringStream.str();
    }
    string measure_body(double minz, double maxz)
    {
        Json::Reader reader;
        Json::Value root;
        Json::Value rt;
        if(flag_weight_set == true)
        {
            rt["weight"] = weight_set;
        }
        if (flag_std_sample)
        {
            cloth_type = "nake";
        }
        // step 4. measure the body
        Json::Value jmeasure, jmeasure_add;
        if(measure_type == "tieshen")
        {
            string body_file = "./results/deformed_rbody.ply";
            bm.loadMeasureBody("./results/deformed_rbody.ply");
            rt["body_model"] = (readFileIntoString("./results/deformed_rbody.ply").c_str());
        }
        else
        {
            string body_file = "./results/body_measure.ply";
            bm.loadMeasureBody("./results/body_measure.ply");
            rt["body_model"] = (readFileIntoString("./results/body_measure.ply").c_str());
        }
        // bm.loadMeasureBody_pcl("./results/scan.ply", body_file.c_str(), "./results/corres_idxes.mat");
        // load the config file
        Json::Value jval;
        Json::Reader reader2;

        // load the corresponding measuring configure file
        string mfolder = "measure_" + measure_type;
        ifstream cf("./data/body_measure/" + mfolder + "/measure_" + measure_type + ".conf");
        cout << "mfolder : " << mfolder << " , " << measure_type << endl;
        if (!reader2.parse(cf, jval))
        {
            cout << "load config file for measurment failed" << endl;
            rt["status"] = "failed";
            rt["details"] = "load config file for measurment failed";
            return (rt.toStyledString());
        }
        Json::Value::Members members;
        members = jval.getMemberNames(); // 获取所有key的值
        int rshoulder_idx = 0, lshoulder_idx = 0;
        bool flag_houyichang2 = false;
        string houyichang_name;
        double normal_ubr = 0.4;
        for (Json::Value::Members::iterator iterMember = members.begin(); iterMember != members.end(); iterMember++) // 遍历每个key
        {
            std::string strKey = *iterMember;
            int rsidx, lsidx;
            if (strKey == "normal_ubr")
            {
                normal_ubr = jval["normal_ubr"].asDouble();
            }
            if (jval[strKey].isArray())
            {
                string pp_name = "./data/body_measure/" + mfolder + "/" + strKey + ".mat";
                string ms_type = jval[strKey][0].asString();
                string ms_name = jval[strKey][1].asString();
                arma::mat tmp;
                tmp.load(pp_name);
                double length = 0;
                cout << "measuring: " << ms_type << "," << strKey << "," << pp_name << endl;
                // for test, save measuring point to file
                bool flag_show = false;
                // if(strKey == "tuiwei")
                {
                    bm.showIndexWithColor("./results/idx" + strKey + ".ply", tmp);
                }
                if (ms_type == "hori_circle")
                {
                    bm.MeasureCircleHori(length, tmp);
                }
                else if (ms_type == "circle")
                {
                    bm.MeasureCircle(length, tmp, flag_show);
                }
                else if (ms_type == "angle")
                {
                    bm.MeasureAngle(length, tmp);
                }
                else if (ms_type == "length")
                {
                    bm.MeasureLength(length, tmp);
                }
                else if (ms_type == "v_length")
                {
                    bm.MeasureLengthVertical(length, tmp);
                }
                else if (ms_type == "h_length_y")
                {
                    bm.MeasureLength_hori_y(length, tmp);
                }
                else if (ms_type == "houyichang2")
                {
                    flag_houyichang2 = true;
                    houyichang_name = ms_name;
                    bm.MeasureHouyichang2(length, tmp);    // 必须有身高数据，返回为上下半身比，等测量身高后再调整为后衣长
                    cout << "ubr is : " << length << endl; // 上下半身比例
                }
                else if (ms_type == "jiankuan")
                {
                    cout << "measure jiankuan" << endl;
                    bm.MeasureLengthJiankuan(length, 0.71, tmp, rsidx, lsidx, "_shoulder.ply");
                    rshoulder_idx = rsidx;
                    lshoulder_idx = lsidx;
                }
                else
                {
                    cout << "measure type unsupported!!! " << ms_type << endl;
                }
                if (ms_type == "angle")
                    jmeasure[ms_name] = round(length);
                else if (ms_type == "houyichang2")
                {
                    jmeasure[ms_name] = length;
                }
                else
                    jmeasure[ms_name] = round(length * 1000) / 10.0;
            }
        }
        double shengao = (maxz - minz);
        jmeasure["身高"] = ceil(shengao * 100);
        cout << "measure finished" << endl;
        rt["left_shoulder_idx"] = lshoulder_idx;
        rt["right_shoulder_idx"] = rshoulder_idx;
        rt["status"] = "succeeded";
        rt["details"] = "call body3D succeeded";
        rt["measures"] = jmeasure;
        rt["time"] = getCurrentTimeStr();
        rt["cloth_type"] = cloth_type;
        rt["measure_type"] = measure_type;
        rt["gender"] = gender;
        Json::StreamWriterBuilder jswBuilder;
        jswBuilder["emitUTF8"] = true;
        std::unique_ptr<Json::StreamWriter> jsWriter(jswBuilder.newStreamWriter());

        std::ostringstream os;
        jsWriter->write(rt, &os);

        ofstream of(spfile_folder + "result.txt");
        of << os.str();
        of.close();
        // end saving

        cout << "config file is: " << cloth_type << endl;
        cout << "End of process @ " << getCurrentTimeStr() << endl;
        // save result for checking

        string cppath = "cp ./results/kn0.ply " + spfile_folder + "kn0.ply";
        auto rtn = system(cppath.c_str());
        cppath = "cp ./examples/final.ply " + spfile_folder + "final.ply";
        rtn = system(cppath.c_str());
        cppath = "cp ./results/scan.ply " + spfile_folder + "scan.ply";
        rtn = system(cppath.c_str());
        cppath = "cp ./results/rbody.ply " + spfile_folder + "rbody.ply";
        rtn = system(cppath.c_str());
        cppath = "cp ./results/body_measure.ply " + spfile_folder + "body_measure.ply";
        rtn = system(cppath.c_str());
        ofstream ff1("log.txt", ios::app);
        ff1 << "end calling the service @ " << getCurrentTimeStr() << endl;
        ff1 << endl;
        ff1.close();
        cout << "end calling the service @ " << getCurrentTimeStr() << endl;
        return os.str();
    }
    void init_bodymeasuer()
    {
        bm.setPathPre("./data/body_measure/");
        pfile = "./examples/final.ply";
        meshFittorMale.setGender("male");
        meshFittorFemale.setGender("female");
        cout << "load male" << endl;
        meshFittorMale.loadTemplate();
        cout << "male model loaded" << endl;
        cout << "load female" << endl;
        meshFittorFemale.loadTemplate();
        cout << "female model loaded" << endl;
        meshFittor = &meshFittorFemale;
    }
#endif
private: 
    double weight_set = 50;
    bool flag_weight_set = false;
};

#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>
#include <thread>
#include <chrono>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/buffer.h>

typedef websocketpp::client<websocketpp::config::asio_client> client;

// Base64 编码函数
std::string base64_encode(const std::vector<unsigned char> &input)
{
    BIO *bio, *b64;
    BUF_MEM *bufferPtr;

    b64 = BIO_new(BIO_f_base64());
    bio = BIO_new(BIO_s_mem());
    bio = BIO_push(b64, bio);

    BIO_set_flags(bio, BIO_FLAGS_BASE64_NO_NL);
    BIO_write(bio, input.data(), input.size());
    BIO_flush(bio);
    BIO_get_mem_ptr(bio, &bufferPtr);
    BIO_set_close(bio, BIO_NOCLOSE);
    BIO_free_all(bio);

    return std::string(bufferPtr->data, bufferPtr->length);
}

// Base64 解码函数
std::vector<unsigned char> base64_decode(const std::string &input)
{
    BIO *bio, *b64;
    std::vector<unsigned char> result(input.size());

    bio = BIO_new_mem_buf(input.c_str(), -1);
    b64 = BIO_new(BIO_f_base64());
    bio = BIO_push(b64, bio);

    BIO_set_flags(bio, BIO_FLAGS_BASE64_NO_NL);
    int decoded_size = BIO_read(bio, result.data(), input.size());
    BIO_free_all(bio);

    result.resize(decoded_size);
    return result;
}

class WebSocketClient
{
public:
    WebSocketClient() : m_done(false), m_connected(false), m_connection_failed(false)
    {
        m_client.clear_access_channels(websocketpp::log::alevel::all);
        m_client.clear_error_channels(websocketpp::log::elevel::all);

        m_client.init_asio();

        m_client.set_open_handler([this](websocketpp::connection_hdl hdl)
                                  {
            m_connected = true;
            m_hdl = hdl;
            std::cout << "WebSocket 连接已打开" << std::endl; });

        m_client.set_message_handler([this](websocketpp::connection_hdl hdl, client::message_ptr msg)
                                     {
            std::cout << "收到服务器消息: " << msg->get_payload() << std::endl;
            m_received_message = msg->get_payload();
            m_done = true; });

        m_client.set_close_handler([this](websocketpp::connection_hdl hdl)
                                   {
            m_connected = false;
            std::cout << "WebSocket 连接已关闭" << std::endl; });

        m_client.set_fail_handler([this](websocketpp::connection_hdl hdl)
                                  {
            auto con = m_client.get_con_from_hdl(hdl);
            std::cerr << "WebSocket 连接失败: " << con->get_ec().message() << std::endl;
            m_connected = false;
            m_connection_failed = true; });
    }

    bool run(const std::string &uri)
    {
        websocketpp::lib::error_code ec;
        client::connection_ptr con = m_client.get_connection(uri, ec);
        if (ec)
        {
            std::cerr << "连接错误: " << ec.message() << std::endl;
            return false;
        }

        m_client.connect(con);

        // 使用独立线程运行客户端
        std::thread client_thread([this]()
                                  { m_client.run(); });

        // 等待连接建立或失败
        while (!m_connected && !m_connection_failed)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (m_connection_failed)
        {
            client_thread.join();
            return false;
        }

        client_thread.detach();
        return true;
    }

    bool send(const std::string &message)
    {
        if (!m_connected)
        {
            std::cout << "WebSocket 未连接，无法发送消息" << std::endl;
            return false;
        }
        websocketpp::lib::error_code ec;
        m_client.send(m_hdl, message, websocketpp::frame::opcode::text, ec);
        if (ec)
        {
            std::cout << "发送消息失败: " << ec.message() << std::endl;
            return false;
        }
        return true;
    }

    void close()
    {
        if (m_connected)
        {
            websocketpp::lib::error_code ec;
            m_client.close(m_hdl, websocketpp::close::status::normal, "Closing connection", ec);
            if (ec)
            {
                std::cout << "关闭连接失败: " << ec.message() << std::endl;
            }
        }
    }

    bool is_done() const { return m_done; }
    bool is_connected() const { return m_connected; }
    const std::string &get_received_message() const { return m_received_message; }

private:
    client m_client;
    websocketpp::connection_hdl m_hdl;
    bool m_done;
    bool m_connected;
    bool m_connection_failed;
    std::string m_received_message;
};

#include <openssl/rsa.h>
#include <openssl/pem.h>
#include <openssl/err.h>
#include <string>
#include <vector>

class RSACrypto
{
public:
    static std::vector<unsigned char> encrypt(const std::string &plain_text, RSA *public_key);
    static std::string decrypt(const std::vector<unsigned char> &cipher_text, RSA *private_key);
};

// 实现 RSACrypto 类的方法
std::vector<unsigned char> RSACrypto::encrypt(const std::string &plain_text, RSA *public_key)
{
    std::vector<unsigned char> encrypted(RSA_size(public_key));
    int encrypted_length = RSA_public_encrypt(plain_text.length(),
                                              reinterpret_cast<const unsigned char *>(plain_text.c_str()),
                                              encrypted.data(),
                                              public_key,
                                              RSA_PKCS1_OAEP_PADDING);
    if (encrypted_length == -1)
    {
        throw std::runtime_error("加密失败");
    }
    encrypted.resize(encrypted_length);
    return encrypted;
}

std::string RSACrypto::decrypt(const std::vector<unsigned char> &cipher_text, RSA *private_key)
{
    std::vector<unsigned char> decrypted(RSA_size(private_key));
    int decrypted_length = RSA_private_decrypt(cipher_text.size(),
                                               cipher_text.data(),
                                               decrypted.data(),
                                               private_key,
                                               RSA_PKCS1_OAEP_PADDING);
    if (decrypted_length == -1)
    {
        throw std::runtime_error("解密失败");
    }
    return std::string(reinterpret_cast<char *>(decrypted.data()), decrypted_length);
}

int main(int argc, char *argv[])
{
    // jianquan
    bool flag_jianquan = false;
if(flag_jianquan){
    OpenSSL_add_all_algorithms();
    ERR_load_crypto_strings();
    std::string publicKeyFile = "./apps/public_key.pem"; // 公钥文件路径
    std::string plain_text = "server_A40_1";             // 原始的plaintext
    std::string local_mac = get_local_mac();             // 获取本机MAC地址

    // 合并MAC地址和原始plaintext
    plain_text = local_mac + "_" + plain_text;

    std::cout << "Combined plaintext: " << plain_text << std::endl;

    std::vector<unsigned char> encrypted; // 加密后的数据
    FILE *public_key_file = fopen(publicKeyFile.c_str(), "rb");
    if (!public_key_file)
    {
        std::cerr << "无法打开公钥文件" << std::endl;
        return 1;
    }
    RSA *public_key = PEM_read_RSAPublicKey(public_key_file, nullptr, nullptr, nullptr);
    fclose(public_key_file);
    if (!public_key)
    {
        std::cerr << "无法读取公钥" << std::endl;
        return 1;
    }
    try
    {
        // 加密
        std::vector<unsigned char> cipher_text = RSACrypto::encrypt(plain_text, public_key);

        // 将加密后的数据转换为Base64字符串
        std::string encoded_cipher = base64_encode(cipher_text);
        std::cout << "Base64编码后的加密数据: " << encoded_cipher << std::endl;

        // 将Base64字符串解码回二进制数据
        std::vector<unsigned char> decoded_cipher = base64_decode(encoded_cipher);

        // 创建 WebSocket 客户端并连接到服务器
        WebSocketClient client;
        if (!client.run("ws://175.6.27.254:7777"))
        {
            std::cerr << "WebSocket 连接失败，程序结束" << std::endl;
            return 1;
        }

        // 发送加密数据到服务器，不进行 Base64 编码
        std::string encoded_message = base64_encode(cipher_text);
        // 添加以下代码来显示解码后的数据（十六进制格式）
        std::vector<unsigned char> decoded_data = base64_decode(encoded_message);
        std::cout << std::dec << std::endl;

        if (!client.send(encoded_message))
        {
            client.close();
            std::cerr << "鉴权失败，无使用权限，错误码：0004" << std::endl;
            return 1;
        }

        // 等待服务器响应
        auto start_time = std::chrono::steady_clock::now();
        while (!client.is_done())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            auto current_time = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count() > 10)
            {
                std::cerr << "鉴权失败，无使用权限，错误码：0003" << std::endl;
                return 1;
            }
        }

        // 获取服务器响应
        std::string server_response = client.get_received_message();

        if (server_response == "ok")
        {
            std::cout << "验证成功：加密和解密操作正确" << std::endl;
        }
        else
        {
            std::cerr << "鉴权失败，无使用权限，错误码：0001" << std::endl;
            return 1;
        }

        // 关闭 WebSocket 连接
        client.close();
    }
    catch (const std::exception &e)
    {
        std::cerr << "鉴权失败，无使用权限，错误码：0002" << std::endl;
        return 1;
    }
    RSA_free(public_key);
    EVP_cleanup();
    ERR_free_strings();
}
    int device = 0;
    cuda::setDevice(device);
    cuda::printShortCudaDeviceInfo(device);

    if (cuda::checkIfPreFermiGPU(device))
        return std::cout << std::endl
                         << "Kinfu is not supported for pre-Fermi GPU architectures, and not built for them by default. Exiting..." << std::endl,
               1;

    KinFuApp app;
#ifdef COMBIN_MS
    std::cout << "init body measure" << std::endl;
    app.init_bodymeasuer();
    cout << "body initialized" << endl;
    // 根据需要调整
    app.measure_type = "qipao";
    app.cloth_type = "tieshen";
    app.gender = "male";
    std::cout << "argc number:" << argc << std::endl;
#endif
    // if(argc >= 2)
    //{
    //     string filename(argv[1]);
    //     std::cout<<app.func(filename)<<endl;
    //     // app.measure_body(0,100);
    //     return 0;
    // }
    if (argc == 2)
    {
        app.port = std::atoi(argv[1]);
        std::cout << "set port to " << app.port << std::endl;
    }
    {
        // executing
        try
        {
            app.execute_ws();
        }
        catch (const std::bad_alloc &e)
        {
            std::cout << "Bad alloc" << e.what() << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cout << "Exception" << e.what() << std::endl;
        }
        catch (...)
        {
            std::cout << "Unknown exception" << std::endl;
        }
        std::cout << "finished" << std::endl;
        return 0;
    }
}
