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
#include "bodymeasurer.h"
#include <unistd.h>
#include <netdb.h>  //gethostbyname
#include <arpa/inet.h>  //ntohl
#include <iostream>
#include <pthread.h>
#include <jsoncpp/json/json.h>
#include "CWebsocketServer.hpp"
using namespace kfusion;
#define COMBIN_MS //if body measurement is combined


static const std::string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";


static inline bool is_base64(unsigned char c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

std::string base64_decode(const std::string& encoded_string) {
    size_t in_len = encoded_string.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::string ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4) {
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

    if (i) {
        for (j = 0; j < i; j++)
            char_array_4[j] = base64_chars.find(char_array_4[j]) & 0xff;

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);

        for (j = 0; (j < i - 1); j++) ret += char_array_3[j];
    }

    return ret;
}

class KinFuApp
{
public:
    int frame_idx = 0;
    // static void KeyboardCallback(const cv::viz::KeyboardEvent& event, void* pthis)
    // {
    //     KinFuApp& kinfu = *static_cast<KinFuApp*>(pthis);
    //     if(event.action != cv::viz::KeyboardEvent::KEY_DOWN)
    //         return;
    //     if(event.code == 't' || event.code == 'T')
    //         kinfu.take_cloud(*kinfu.kinfu_);
    //     if(event.code == 'i' || event.code == 'I')
    //         kinfu.iteractive_mode_ = !kinfu.iteractive_mode_;
    // }

    KinFuApp() : exit_(false),  iteractive_mode_(false), pause_(true)
    {
        KinFuParams params = KinFuParams::default_params();
        kinfu_ = KinFu::Ptr( new KinFu(params) );
        // capture_.setRegistration(true);
        // cv::viz::WCube cube(cv::Vec3d::all(0), cv::Vec3d(params.volume_size), true, cv::viz::Color::apricot());
        // viz.showWidget("cube", cube, params.volume_pose);
        // viz.showWidget("coor", cv::viz::WCoordinateSystem(0.1));
        // viz.registerKeyboardCallback(KeyboardCallback, this);
    }

    void show_depth(const cv::Mat& depth)
    {
        cv::Mat display;
        //cv::normalize(depth, display, 0, 255, cv::NORM_MINMAX, CV_8U);
        depth.convertTo(display, CV_8U, 255.0/65535);
        cv::imshow("Depth", display);
    }

    void show_raycasted(KinFu& kinfu)
    {
        const int mode = 0;
        if (iteractive_mode_)
            ;//kinfu.renderImage(view_device_, viz.getViewerPose(), mode);
        else
            kinfu.renderImage(view_device_, mode);

        view_host_.create(view_device_.rows(), view_device_.cols(), CV_8UC4);
        points_host_.create(view_device_.rows(), view_device_.cols(), CV_32FC4);
        // kinfu.getPoints(points_host_);
        // std::stringstream ss;
        // ss<<"./results/rst"<<frame_idx<<".ply";
        // kinfu.toPly(points_host_,ss.str());
        view_device_.download(view_host_.ptr<void>(), view_host_.step);
        cv::imshow("Scene", view_host_);
    }

    void take_cloud(KinFu& kinfu)
    {
        cuda::DeviceArray<Point> cloud = kinfu.tsdf().fetchCloud(cloud_buffer);
        cv::Mat cloud_host(1, (int)cloud.size(), CV_32FC4);
        cloud.download(cloud_host.ptr<Point>());
        // viz.showWidget("cloud", cv::viz::WCloud(cloud_host));
        //viz.showWidget("cloud", cv::viz::WPaintedCloud(cloud_host));
    }
    // #include <librealsense2/rs.hpp>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <unistd.h>
    #include <string.h>
    #include <thread>
    #include <fstream>
    bool execute_ws()
    {
        CWSServer ws;
        ws.set_port(9099);
        thread t_ws(bind(&CWSServer::excute,&ws));
        bool flag_got = false;
        int fid = 0;
        KinFu& kinfu = *kinfu_;
        cv::Mat depth, image;
        double time_ms = 0;
        bool has_image = false;
        pause_ = false;
        while (true)
        {
            std::cout<<"try to achieve msg"<<std::endl;
            auto a = ws.archieve_message(flag_got);
            if(flag_got)
            {
                
                //start to process the message
                // std::cout<<a.msg_str<<std::endl;
                std::string ws_str = base64_decode(a.msg->get_payload());
                std::cout<<"image received: "<<ws_str.size()<<std::endl;
                if(ws_str.size()<100)
                {
                    std::cout<<"msg: "<<ws_str<<std::endl;
                }
                // if(ws_str == "finished")
                if(ws_str.size()< 100)
                {
                    std::cout<<"finished and try to measure"<<std::endl;
                    //save final cloud to file
                    cuda::DeviceArray<Point> cloud = kinfu.tsdf().fetchCloud(cloud_buffer);
                    kinfu.tsdf().fetchNormals(cloud,normal_buffer);
                    //
                    cv::Mat cloud_host(1, (int)cloud.size(), CV_32FC4);
                    cloud.download(cloud_host.ptr<Point>());
                    //
                    cv::Mat normal_host(1, (int)cloud.size(), CV_32FC4);
                    normal_buffer.download(normal_host.ptr<Point>());
                    
                    #ifdef COMBIN_MS
                    //save to file for measurement
                    std::stringstream ss;
                    ss<<pfile;
                    kinfu.toPlyColorFilter(cloud_host, normal_host, ss.str(),255,0,0);
                    //start measurement

                    auto rst = func(readFileIntoString((char *)pfile.c_str()));
                    ws.send_msg(a.hdl,rst);
                    #endif
                    kinfu.tsdf().clear();

                }
                else{
                    std::cout<<"base 64 decoded"<<std::endl;
                    std::vector<unsigned char> img_vec(ws_str.begin(), ws_str.end());
                    std::cout<<"decode png"<<std::endl;
                    cv::Mat depth = cv::imdecode(img_vec, cv::IMREAD_ANYDEPTH);
                    depth = depth /4;
                    ////////////////////////START//////////////////////////
                    // user specified code, for test to filter the point cloud
                    for (size_t i = 0; i < depth.rows; i++)
                    {
                        for (size_t j = 0; j < depth.cols; j++)
                        {
                            if(depth.at<ushort>(i,j)>2000)
                            {
                                depth.at<ushort>(i,j) = 0;
                            }
                        }
                    }
                    // cv::Rect maskroi(0,0,200,720);
                    // depth(maskroi) = 0;
                    ////////////////////////END//////////////////////////
                    depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);
                    // depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);
                    {
                        SampledScopeTime fps(time_ms); (void)fps;
                        has_image = kinfu(depth_device_);
                    }

                    if (has_image)
                        show_raycasted(kinfu);

                    // show_depth(depth);
                    cv::imshow("Image", depth);
                    int key = cv::waitKey(pause_ ? 0 : 3);

                    switch(key)
                    {
                    case 't': case 'T' : take_cloud(kinfu); break;
                    case 'i': case 'I' : iteractive_mode_ = !iteractive_mode_; break;
                    case 's':pause_ = false;break;
                    case 27: exit_ = true; break;
                    case 'p': pause_ = !pause_; break;
                    }
                    std::cout<<"image received"<<std::endl;
                }
            }
            else{
                std::cout<<"achieve msg failed"<<std::endl;
            }
        }
        
        return true;
    }

    bool execute()
    {
        KinFu& kinfu = *kinfu_;
        cv::Mat depth, image;
        double time_ms = 0;
        bool has_image = false;

        std::vector<cv::String> depths;             // store paths,
        std::vector<cv::String> images;             // store paths,

        cv::glob("./data_kinfu/rotperson_1/depth", depths);
        cv::glob("./data_kinfu/rotperson_1/color", images);

        std::sort(depths.begin(), depths.end());
        std::sort(images.begin(), images.end());

        pause_ = true;
        for (int i = 200; i < depths.size() && !exit_ ; ++i)
        { 
            // if(i>200)
            //     exit_ = true;
            frame_idx = i;
            std::cout<<"frame: "<<i<<std::endl;
            // bool has_frame = capture_.grab(depth, image);
            image = cv::imread(images[i], cv::IMREAD_COLOR);
            depth = cv::imread(depths[i], cv::IMREAD_ANYDEPTH);
            depth = depth /4;
            ////////////////////////START//////////////////////////
            // user specified code, for test to filter the point cloud
            for (size_t i = 0; i < depth.rows; i++)
            {
                for (size_t j = 0; j < depth.cols; j++)
                {
                    if(depth.at<ushort>(i,j)>2000)
                    {
                        depth.at<ushort>(i,j) = 0;
                    }
                }
            }
            // cv::Rect maskroi(0,0,200,720);
            // depth(maskroi) = 0;
            ////////////////////////END//////////////////////////
            depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);
            // depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);
            {
                SampledScopeTime fps(time_ms); (void)fps;
                has_image = kinfu(depth_device_);
            }

            if (has_image)
                show_raycasted(kinfu);

            // show_depth(depth);
            cv::imshow("Image", image);

            // if (!iteractive_mode_)
            //     viz.setViewerPose(kinfu.getCameraPose());

            int key = cv::waitKey(pause_ ? 0 : 3);

            switch(key)
            {
            case 't': case 'T' : take_cloud(kinfu); break;
            case 'i': case 'I' : iteractive_mode_ = !iteractive_mode_; break;
            case 's':pause_ = false;break;
            case 27: exit_ = true; break;
            case 'p': pause_ = !pause_; break;
            }

            //exit_ = exit_ || i > 100;
            // viz.spinOnce(3, true);
            std::cout<<"finish frame"<<std::endl;
        }
        //save final cloud to file
        cuda::DeviceArray<Point> cloud = kinfu.tsdf().fetchCloud(cloud_buffer);
        kinfu.tsdf().fetchNormals(cloud,normal_buffer);
        //
        cv::Mat cloud_host(1, (int)cloud.size(), CV_32FC4);
        cloud.download(cloud_host.ptr<Point>());
        //
        cv::Mat normal_host(1, (int)cloud.size(), CV_32FC4);
        normal_buffer.download(normal_host.ptr<Point>());
        
        #ifdef COMBIN_MS
        //save to file for measurement
        std::stringstream ss;
        ss<<pfile;
        kinfu.toPlyColorFilter(cloud_host, normal_host, ss.str(),255,0,0);
        //start measurement

        func(readFileIntoString((char *)pfile.c_str()));
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
    string readFileIntoString(char * filename)
    {
        ifstream ifile(filename);
        //将文件读入到ostringstream对象buf中
        ostringstream buf;
        char ch;
        while(buf&&ifile.get(ch))
        buf.put(ch);
        //返回与流对象buf关联的字符串
        return buf.str();
    }

    static string  getCurrentTimeStr()
    {
        time_t t = time(NULL);
        char ch[64] = {0};
        strftime(ch, sizeof(ch) - 1, "%Y-%m-%d %H:%M:%S", localtime(&t));     //年-月-日 时-分-秒
        return ch;
    }
    string func(string param_json)
    {
        ofstream ff("log.txt",ios::app);
        ff<<"calling the service @ "<<getCurrentTimeStr()<<endl;
        ff.close();
        string zhuozhuang_type = "jinshen";
        Json::Reader reader;
        Json::Value root;
        Json::Value rt;
        // cout<<"in func"<<endl;
        // rt["status"]="test";
        // rt["details"]="test, return without processing";
        // return(rt.toStyledString());
        string gender = "male";
        {
            if(gender=="male")
            {
                meshFittor = &meshFittorMale;
            }
            else
            {
                meshFittor = &meshFittorFemale;
            }
        }

        // zhuozhuang_type = "tieshen";

        cout<<"step 1. load ply file:"<<pfile<<endl;
        //step 1. load ply file
        Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity ();
        transformation_matrix (0, 0) = 1;
        transformation_matrix (1, 1) = 1;
        transformation_matrix (2, 2) = 1;
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_orig (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        if(pcl::io::loadPLYFile<pcl::PointXYZRGBNormal> ("./examples/final.ply", *cloud_orig) == -1)
        {
            PCL_ERROR("Could not read file \n");
        }
        if(cloud_orig->size()<10000)
        {
            return "{\"error\":\"points not enough\"}";
        }
        std::cout<<"ply file loaded, try to filter the data"<<std::endl;
        pcl::transformPointCloud (*cloud_orig, *cloud, transformation_matrix);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        //step 2 filter the outlier points
        cout<<"set transformed"<<endl;
        pcl::RadiusOutlierRemoval<pcl::PointXYZRGBNormal> sor;
        sor.setInputCloud(cloud);
        sor.setRadiusSearch(0.05);
        sor.setMinNeighborsInRadius(20);
        sor.setNegative(false);
        cout<<"start filter"<<endl;
        sor.filter(*cloud_filtered);

        pcl::io::savePLYFile("./results/aft_filter.ply",*cloud_filtered);
    
        pcl::PointCloud<pcl::PointXYZRGBNormal> scan;
        scan = * cloud_filtered;
        //step 3. fit the model
        // scan = *cloud;
        double maxz=-10000000,minz=10000000;
        for(int i=0;i<scan.points.size();i++)
        {
            double z = scan.points[i].x;
            if(z>maxz)
                maxz = z;
            if(z<minz)
                minz = z;
        }
        pcl::io::savePLYFile("./results/scan.ply",scan);
        Json::Value val_weights;
        ifstream rcf("./data/weights_"+zhuozhuang_type+".conf");
        if(!reader.parse(rcf,val_weights))
        {
            cout<<"===================================================load config file for measurment failed"<<endl;
        }
        else
        {
            cout<<"config file is: "<<zhuozhuang_type<<endl;
            meshFittor->setWeights(val_weights["weight_out"].asDouble(),val_weights["weight_in"].asDouble() ,val_weights["weight_in_margin"].asDouble(), val_weights["in_thick"].asDouble() );
            cout<<"===============================================weights seted"<<endl;
        }
        cout<<"reset parameters"<<endl;
        meshFittor->resetParameters();
        cout<<"start mainprocess"<<endl;
        meshFittor->mainProcess(scan);
        cout<<"start measure"<<endl;
        return measure_body(minz, maxz);
        
    }
    string measure_body(double minz, double maxz)
    {
        Json::Reader reader;
        Json::Value root;
        Json::Value rt;
        string zhuozhuang_type="jinshen";
        //step 4. measure the body
        Json::Value jmeasure,jmeasure_add;
        // bm.loadMeasureBody("../results/rbody.ply");
        bm.loadMeasureBody("../results/male_init_tempbody.ply");
        //load the config file
        Json::Value jval;
        Json::Reader reader2;
        string measure_type = "qipao";

        //load the corresponding measuring configure file
        string mfolder = "measure_"+measure_type;
        ifstream cf("./data/body_measure/"+mfolder+"/measure_"+measure_type+".conf");
        cout<<"mfolder : "<<mfolder<<" , "<<measure_type<<endl;
        if(!reader2.parse(cf,jval))
        {
            cout<<"load config file for measurment failed"<<endl;
            rt["status"]="failed";
            rt["details"]="load config file for measurment failed";
            return(rt.toStyledString());
        }
        Json::Value::Members members;
        members = jval.getMemberNames();   // 获取所有key的值
        int rshoulder_idx=0, lshoulder_idx=0;
        bool flag_houyichang2 = false;
        string houyichang_name;
        double normal_ubr = 0.4;
        for (Json::Value::Members::iterator iterMember = members.begin(); iterMember != members.end(); iterMember++)   // 遍历每个key
        {
            std::string strKey = *iterMember;
            int rsidx, lsidx;
            if(strKey == "normal_ubr")
            {
                normal_ubr = jval["normal_ubr"].asDouble();
            }
            if(jval[strKey].isArray())
            {
                string pp_name = "./data/body_measure/"+mfolder+"/"+strKey+".mat";
                string ms_type = jval[strKey][0].asString();
                string ms_name = jval[strKey][1].asString();
                arma::mat tmp;
                tmp.load(pp_name);
                double length = 0;
                cout<<"measuring: "<<ms_type<<","<<strKey<<endl;
                //for test, save measuring point to file
                bool flag_show = false;
                if(strKey == "tuiwei")
                {
                    bm.showIndexWithColor("../results/idx"+strKey+".ply",tmp);
                    flag_show = true;
                    cout<<"measure type of "<<strKey<<": "<<ms_type<<endl;
                }
                if(ms_type == "hori_circle")
                {
                    bm.MeasureCircleHori(length,tmp);
                }
                else if(ms_type=="circle")
                {
                    bm.MeasureCircle(length,tmp,flag_show);
                }
                else if(ms_type == "angle")
                {
                    bm.MeasureAngle(length,tmp);
                }
                else if (ms_type == "length")
                {
                    bm.MeasureLength(length,tmp);
                }
                else if (ms_type == "v_length")
                {
                    bm.MeasureLengthVertical(length,tmp);
                }
                else if (ms_type == "h_length_y")
                {
                    bm.MeasureLength_hori_y(length,tmp);
                }
                else if (ms_type == "houyichang2")
                {
                    flag_houyichang2 = true;
                    houyichang_name = ms_name;
                    bm.MeasureHouyichang2(length,tmp);//必须有身高数据，返回为上下半身比，等测量身高后再调整为后衣长
                    cout<<"ubr is : "<<length<<endl;//上下半身比例
                }
                else if (ms_type == "jiankuan")
                {
                    cout<<"measure jiankuan"<<endl;
                    bm.MeasureLengthJiankuan(length,0.71,tmp,rsidx,lsidx,"_shoulder.ply");
                    rshoulder_idx = rsidx;
                    lshoulder_idx = lsidx;
                }
                else{
                    cout<<"measure type unsupported!!! "<<ms_type<<endl;
                }
                if(ms_type == "angle")
                    jmeasure[ms_name]=round(length);
                else if(ms_type == "houyichang2")
                {
                    jmeasure[ms_name]=length;
                }
                else
                    jmeasure[ms_name]=round(length*1000)/10.0;
            }
        }
        cout<<"measure finished"<<endl;
        rt["left_shoulder_idx"]=lshoulder_idx;
        rt["right_shoulder_idx"]=rshoulder_idx;
        rt["status"]="succeeded";
        rt["details"]="call body3D succeeded";
        double shengao =(maxz-minz);
        double ratio = 1.0;//bm.getHeightRatio();
        if(flag_houyichang2)
        {
            double ratio_ubr = (jmeasure[houyichang_name].asDouble()/normal_ubr);
            if(ratio_ubr > 1.1)
            {
                ratio_ubr = 1.1;
            }
            if(ratio_ubr < 0.9)
            {
                ratio_ubr = 0.9;
            }
            ratio_ubr = (ratio_ubr-1.0)/2+1.0;
            double hl=(shengao*100*0.4738-5.8165)*ratio_ubr;
            jmeasure[houyichang_name] = round(hl*10)/10;
        }
        if(ratio <1.0)
            ratio = 1.0;
        jmeasure["身高"]=ceil(shengao*100);

        // if(jmeasure["身高"].isDouble())
        // {
        //     // cout<<"模型测量身高是:"<<jmeasure["身高"]<<endl;
        //     jmeasure["身高"] = round(shengao * ratio*1000)/10.0;
        // }
        cout<<jmeasure.toStyledString()<<endl;
        return jmeasure.toStyledString();
        //aditional measure
        mfolder = "measure_chenshan_all";
        ifstream cfa("./data/body_measure/"+mfolder+"/measure_chenshan_xx"+".conf");
        if(!reader2.parse(cfa,jval))
        {
            cout<<"load config file for measurment failed"<<endl;
            rt["status"]="failed";
            rt["details"]="load config file for additional measurment failed";
            return(rt.toStyledString());
        }
        members = jval.getMemberNames();   // 获取所有key的值
        for (Json::Value::Members::iterator iterMember = members.begin(); iterMember != members.end(); iterMember++)   // 遍历每个key
        {
            std::string strKey = *iterMember;
            int rsidx, lsidx;
            if(jval[strKey].isArray())
            {
                string pp_name = "./data/body_measure/"+mfolder+"/"+strKey+".mat";
                string ms_type = jval[strKey][0].asString();
                string ms_name = jval[strKey][1].asString();
                arma::mat tmp;
                tmp.load(pp_name);
                double length = 0;
                cout<<"measure type: "<<ms_type<<endl;
                if(ms_type == "hori_circle")
                {
                    bm.MeasureCircleHori(length,tmp);
                }
                else if(ms_type=="circle")
                {
                    bm.MeasureCircle(length,tmp);
                }
                else if(ms_type == "angle")
                {
                    bm.MeasureAngle(length,tmp);
                }
                else if (ms_type == "length")
                {
                    bm.MeasureLength(length,tmp);
                }
                else if (ms_type == "v_length")
                {
                    bm.MeasureLengthVertical(length,tmp);
                }
                else if (ms_type == "jiankuan")
                {
                    cout<<"measure jiankuan"<<endl;
                    bm.MeasureLengthJiankuan(length,0.71,tmp,rsidx,lsidx,"_shoulder.ply");
                }
                else{
                    cout<<"measure type unsupported!!! "<<ms_type<<endl;
                }
                if(ms_type == "angle")
                    jmeasure_add[ms_name]=round(length);
                else
                    jmeasure_add[ms_name]=round(length*1000)/10.0;
            }
        }
        //add qianyaojie,houyaojie to measure
        jmeasure["前腰节长"]=jmeasure_add["前腰节长"].asDouble();
        jmeasure["后腰节长"]=jmeasure_add["后腰节长"].asDouble();
        rt["measures"].append(jmeasure);
        cout<<rt["measures"].toStyledString()<<endl;
        //save measure results to the check results for  checking
        struct timeval tv;
        gettimeofday(&tv,NULL);
        stringstream ss;
        ss<<"./check_results/body_"<<tv.tv_sec;
        string spfile_name = ss.str();
        ofstream of(spfile_name+".txt");
        of<<zhuozhuang_type<<endl;
        of<<rt.toStyledString();
        of.close();
        //end saving
        rt["model"]=readFileIntoString("./results/rbody.ply");
        rt["kn"]=readFileIntoString("./results/kn0.ply");
        rt["ptcloud"] = readFileIntoString("./results/aft_filter.ply");
        // rt["test_load"]=readFileIntoString("./results/load.ply");
        //
        rt["measures_additional"].append(jmeasure_add);
        cout<<rt["measures_additional"].toStyledString()<<endl;
        cout<<"config file is: "<<zhuozhuang_type<<endl;
        // system("python ./scripts/upload_data.py");
        cout<<"End of process @ "<<getCurrentTimeStr()<<endl;
        //save result for checking

        string cppath ="cp ./results/kn0.ply "+ spfile_name+"_kn0.ply";
        system(cppath.c_str());
        cppath = "cp ./results/aft_filter.ply "+ spfile_name+"_ptcloud.ply";
        system(cppath.c_str());
        ofstream ff1("log.txt",ios::app);
        ff1<<"end calling the service @ "<<getCurrentTimeStr()<<endl;
        ff1<<endl;
        ff1.close();
        return rt.toStyledString();

    }
    void init_bodymeasuer()
    {
        bm.setPathPre("./data/body_measure/");
        // cloud_filtered = std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBNormal> >(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pfile = "./examples/final.ply";
        meshFittorMale.setGender("male");
        meshFittorFemale.setGender("female");
        cout<<"load male"<<endl;
        meshFittorMale.loadTemplate();
        cout<<"male model loaded"<<endl;
        cout<<"load female"<<endl;
        meshFittorFemale.loadTemplate();
        cout<<"female model loaded"<<endl;
        meshFittor = &meshFittorFemale;
    }
    #endif
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main (int argc, char* argv[])
{
    // //test ply loading
    // pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_orig (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    // pcl::io::loadPLYFile<pcl::PointXYZRGBNormal> ("./examples/final1.ply", *cloud_orig);
    // cout<<"ply loaded"<<endl;
    // return 0;
    int device = 0;
    cuda::setDevice (device);
    cuda::printShortCudaDeviceInfo (device);

    if(cuda::checkIfPreFermiGPU(device))
        return std::cout << std::endl << "Kinfu is not supported for pre-Fermi GPU architectures, and not built for them by default. Exiting..." << std::endl, 1;

    KinFuApp app;
    #ifdef COMBIN_MS
    std::cout<<"init body measure"<<std::endl;
    app.init_bodymeasuer();
    cout<<"body initialized"<<endl;
    #endif
    cout<<app.measure_body(0,175)<<endl;
    return 0;
    // executing
    try { app.execute_ws(); }
    catch (const std::bad_alloc& /*e*/) { std::cout << "Bad alloc" << std::endl; }
    catch (const std::exception& /*e*/) { std::cout << "Exception" << std::endl; }
    std::cout<<"finished"<<std::endl;
    return 0;
}
