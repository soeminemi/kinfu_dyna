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
bool flag_std_sample = false;
bool flag_show_image = false;
static const std::string base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";


static inline bool is_base64(unsigned char c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

unsigned char* base64_encode(const char* str0)
{
	unsigned char* str = (unsigned char*)str0;	//转为unsigned char无符号,移位操作时可以防止错误
	long len;				//base64处理后的字符串长度
	long str_len;			//源字符串长度
	long flag;				//用于标识模3后的余数
	unsigned char* res;		//返回的字符串
	str_len = strlen((const char*)str);
	switch (str_len % 3)	//判断模3的余数
	{
	case 0:flag = 0; len = str_len / 3 * 4; break;
	case 1:flag = 1; len = (str_len / 3 + 1) * 4; break;
	case 2:flag = 2; len = (str_len / 3 + 1) * 4; break;
	}
	res = (unsigned char*)malloc(sizeof(unsigned char) * len + 1);
	for (int i = 0, j = 0; j < str_len - flag; j += 3, i += 4)//先处理整除部分
	{
		//注意&运算和位移运算的优先级,是先位移后与或非
		res[i] = base64_chars[str[j] >> 2];
		res[i + 1] = base64_chars[(str[j] & 0x3) << 4 | str[j + 1] >> 4];
		res[i + 2] = base64_chars[(str[j + 1] & 0xf) << 2 | (str[j + 2] >> 6)];
		res[i + 3] = base64_chars[str[j + 2] & 0x3f];
	}
	//不满足被三整除时,要矫正
	switch (flag)
	{
	case 0:break;	//满足时直接退出
	case 1:res[len - 4] = base64_chars[str[str_len - 1] >> 2];	//只剩一个字符时,右移两位得到高六位
		res[len - 3] = base64_chars[(str[str_len - 1] & 0x3) << 4];//获得低二位再右移四位,自动补0
		res[len - 2] = res[len - 1] = '='; break;				//最后两个补=
	case 2:
		res[len - 4] = base64_chars[str[str_len - 2] >> 2];				//剩两个字符时,右移两位得高六位
		res[len - 3] = base64_chars[(str[str_len - 2] & 0x3) << 4 | str[str_len - 1] >> 4];	//第一个字符低二位和第二个字符高四位
		res[len - 2] = base64_chars[(str[str_len - 1] & 0xf) << 2];	//第二个字符低四位,左移两位自动补0
		res[len - 1] = '=';											//最后一个补=
		break;
	}
	res[len] = '\0';	//补上字符串结束标识
	return res;
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
    string cloth_type="tieshen";
    string measure_type = "qipao";
    string gender = "male";
    string device_type = "kinect";
    string spfile_folder = "./check_results/body_default/";
    string color_folder = "./check_results/body_default/colors";
    string depth_folder = "./check_results/body_default/depths";
    KinFuApp() : exit_(false),  iteractive_mode_(false), pause_(true)
    {
        KinFuParams params = KinFuParams::default_params();
        if(device_type == "kinect")
        {
            params.intr = Intr(365.3566f, 365.3566f, 261.4155f, 206.6168f);
            params.cols = 512;  //pixels
            params.rows = 414;  //pixels
        }
        else if(device_type == "realsense")
        {
            params.cols = 1280;  //pixels
            params.rows = 780;  //pixels
            params.intr = Intr(898.033f, 898.745f, 653.17f, 353.58f);
        }
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
        if(flag_show_image)
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
        bool flag_started = false;
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
        Json::Value jv;
        Json::Reader jreader;
        while (true)
        {
            std::cout<<"try to achieve msg"<<std::endl;
            auto a = ws.archieve_message(flag_got);
            if(flag_got)
            {
                
                //start to process the message
                // std::cout<<a.msg_str<<std::endl;
                auto msg = a.msg->get_payload();
                // std::cout<<"msg received:"<<msg<<std::endl;
                jreader.parse(msg, jv);
                if(jv["ack"].isString())
                {
                    struct timeval tv;
                    gettimeofday(&tv,NULL);
                    //save result and files to sample
                    stringstream ss;
                    ss<<tv.tv_sec;
                    ws.send_msg(a.hdl,ss.str());
                    continue;
                }
                if(jv["cmd"].asString() == "finish")
                {
                    if(jv["gender"].isString())
                    {
                        if(jv["gender"].asString() == "male")
                            meshFittor = &meshFittorMale;
                        else
                            meshFittor = &meshFittorFemale;
                    }
                    if(jv["measure_type"].isString())
                    {
                        measure_type = jv["measure_type"].asString();
                    }
                    if(jv["cloth_type"].isString())
                    {
                        cloth_type = jv["cloth_type"].asString();
                    }
                    flag_started = false;
                    std::cout<<"finished and try to measure"<<std::endl;
                    //save final cloud to file
                    cuda::DeviceArray<Point> cloud = kinfu.tsdf().fetchCloud(cloud_buffer);
                    if(cloud.size() > 0)
                    {
                        std::cout<<"try to fetch normals"<<std::endl;
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
                        auto rst = func(pfile);
                        ws.send_msg(a.hdl,rst);
                        #endif
                    }
                    else{
                        
                        //for test 
                        Json::Value rt;
                        rt["error"] = "points not enough";
                        Json::StreamWriterBuilder jswBuilder;
                        jswBuilder["emitUTF8"] = true;
                        std::unique_ptr<Json::StreamWriter>jsWriter(jswBuilder.newStreamWriter());

                        std::ostringstream os;
                        jsWriter->write(rt, &os);
                        ws.send_msg(a.hdl,os.str());
                    }
                   
                    kinfu.reset();

                }
                else{
                    if(flag_started == false)
                    {
                        struct timeval tv;
                        gettimeofday(&tv,NULL);
                        //save result and files to sample
                        stringstream ss;
                        ss<<"./check_results/body_"<<tv.tv_sec<<"/";
                        spfile_folder = ss.str();
                        string cmd_mkdir = "mkdir -p "+spfile_folder;
                        system(cmd_mkdir.c_str());
                        color_folder = spfile_folder+"colors/";
                        depth_folder = spfile_folder + "depths/";
                        system(("mkdir -p "+color_folder).c_str());
                        system(("mkdir -p "+depth_folder).c_str());
                    }
                    flag_started = true;
                    if(jv["img_type"].asString()=="color")
                    {
                        std::string ws_str = base64_decode(jv["data"].asString());
                        std::vector<unsigned char> img_vec(ws_str.begin(), ws_str.end());
                        std::cout<<"decode png"<<std::endl;
                        cv::Mat color = cv::imdecode(img_vec, cv::IMREAD_COLOR);
                        cv::imwrite(color_folder+jv["frame_id"].asString()+".jpg",color);
                    }
                    else
                    {
                        std::string ws_str = base64_decode(jv["data"].asString());
                        std::cout<<"base 64 decoded"<<std::endl;
                        std::vector<unsigned char> img_vec(ws_str.begin(), ws_str.end());
                        std::cout<<"decode png"<<std::endl;
                        cv::Mat depth = cv::imdecode(img_vec, cv::IMREAD_ANYDEPTH);
                        cv::imwrite(depth_folder+jv["frame_id"].asString()+".png",depth);
                        if(device_type == "realsense")
                        {
                            depth = depth /4;
                        }
                        else if(device_type == "kinect")
                        {
                            depth = depth;
                        }
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
                        if(flag_show_image)
                        {
                            cv::imshow("Image", depth);
                            int key = cv::waitKey(pause_ ? 0 : 3);

                        // switch(key)
                        // {
                        // case 't': case 'T' : take_cloud(kinfu); break;
                        // case 'i': case 'I' : iteractive_mode_ = !iteractive_mode_; break;
                        // case 's':pause_ = false;break;
                        // case 27: exit_ = true; break;
                        // case 'p': pause_ = !pause_; break;
                        // }
                        }
                    }
                    std::cout<<"image received: "<<jv["img_type"].asString()<<std::endl;
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
    string func(string bodypath)
    {
        auto start = std::chrono::high_resolution_clock::now();
        ofstream ff("log.txt",ios::app);
        ff<<"calling the service @ "<<getCurrentTimeStr()<<endl;
        ff.close();
        if(flag_std_sample)
        {
            cloth_type="nake";
        }
        Json::Reader reader;
        Json::Value root;
        Json::Value rt;
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

        // cloth_type = "lvekuansong";

        cout<<"step 1. load ply file:"<<pfile<<endl;
        //step 1. load ply file
        Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Zero ();
        if(flag_std_sample ==  true)
        {
            transformation_matrix (0, 0) = 1;
            transformation_matrix (1, 1) = 1;
            transformation_matrix (2, 2) = 1;
            transformation_matrix (3, 3) = 1;
        }
        else{
            transformation_matrix (0, 1) = -1;
            transformation_matrix (1, 0) = -1;
            transformation_matrix (2, 2) = -1;
            transformation_matrix (3, 3) = 1;
        }
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_orig (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        if(pcl::io::loadPLYFile<pcl::PointXYZRGBNormal> (bodypath, *cloud_orig) == -1)
        {
            PCL_ERROR("Could not read file \n");
        }
        
        if(cloud_orig->size()<10000)
        {
            return "{\"error\":\"points not enough\"}";
        }
        std::cout<<"ply file loaded, try to filter the data"<<std::endl;
        pcl::transformPointCloudWithNormals (*cloud_orig, *cloud, transformation_matrix);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        // for(size_t i = 0;i<cloud->size();i++)
        // {
        //     pcl::PointXYZRGBNormal &tp = (*cloud)[i];
        //     tp.normal_x = -tp.normal_x;
        //     tp.normal_y = -tp.normal_y;
        //     tp.normal_z = -tp.normal_z;
        // }
        
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
            double z = scan.points[i].y;
            if(z>maxz)
                maxz = z;
            if(z<minz)
                minz = z;
        }
        pcl::io::savePLYFile("./results/scan.ply",scan);
        Json::Value val_weights;
        ifstream rcf("./data/weights_"+cloth_type+".conf");
        if(!reader.parse(rcf,val_weights))
        {
            cout<<"===================================================load config file for measurment failed"<<endl;
        }
        else
        {
            cout<<"config file is: "<<cloth_type<<endl;
            meshFittor->setWeights(val_weights["weight_out"].asDouble(),val_weights["weight_in"].asDouble() ,val_weights["weight_in_margin"].asDouble(), val_weights["in_thick"].asDouble() );
            cout<<"===============================================weights seted"<<endl;
        }
        cout<<"reset parameters"<<endl;
        meshFittor->resetParameters();
        cout<<"start mainprocess"<<endl;
        meshFittor->mainProcess(scan);
        cout<<"start measure"<<endl;
        auto rst =  measure_body(minz, maxz);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        cout<<"time used for optimization: "<<elapsed.count()<<endl;
        return rst;
        
    }
    std::string readFileToString(std::string filePath) 
    {
        std::ifstream fileStream(filePath);
        if (!fileStream.is_open()) {
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
        
        if(flag_std_sample)
        {
            cloth_type="nake";
        }
        //step 4. measure the body
        Json::Value jmeasure,jmeasure_add;
        bm.loadMeasureBody("./results/rbody.ply");
        rt["body_model"] = (readFileIntoString("./results/rbody.ply").c_str());
        // bm.loadMeasureBody_pcl("./results/scan.ply", "./results/rbody.ply", "./results/corres_idxes.mat");
        //load the config file
        Json::Value jval;
        Json::Reader reader2;

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
                cout<<"measuring: "<<ms_type<<","<<strKey<<","<<pp_name<<endl;
                //for test, save measuring point to file
                bool flag_show = false;
                // if(strKey == "tuiwei")
                {
                    bm.showIndexWithColor("./results/idx"+strKey+".ply",tmp);
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
        double shengao =(maxz-minz);
        jmeasure["身高"]=ceil(shengao*100);
        cout<<"measure finished"<<endl;
        rt["left_shoulder_idx"]=lshoulder_idx;
        rt["right_shoulder_idx"]=rshoulder_idx;
        rt["status"]="succeeded";
        rt["details"]="call body3D succeeded";
        rt["measures"] = jmeasure;
        rt["time"] = getCurrentTimeStr();
        rt["cloth_type"] = cloth_type;
        rt["measure_type"] = measure_type;
        rt["gender"] = gender;
        Json::StreamWriterBuilder jswBuilder;
        jswBuilder["emitUTF8"] = true;
        std::unique_ptr<Json::StreamWriter>jsWriter(jswBuilder.newStreamWriter());

        std::ostringstream os;
        jsWriter->write(rt, &os);

        ofstream of(spfile_folder+"result.txt");
        of<<os.str();
        of.close();
        //end saving
        
        cout<<"config file is: "<<cloth_type<<endl;
        cout<<"End of process @ "<<getCurrentTimeStr()<<endl;
        //save result for checking

        string cppath ="cp ./results/kn0.ply "+ spfile_folder+"kn0.ply";
        system(cppath.c_str());
        cppath = "cp ./examples/final.ply "+ spfile_folder+"final.ply";
        system(cppath.c_str());
        ofstream ff1("log.txt",ios::app);
        ff1<<"end calling the service @ "<<getCurrentTimeStr()<<endl;
        ff1<<endl;
        ff1.close();
        cout<<"end calling the service @ "<<getCurrentTimeStr()<<endl;

        return os.str();
    }
    void init_bodymeasuer()
    {
        bm.setPathPre("./data/body_measure/");
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
    cout<<"usage: --test for test, follow with ply file path"<<endl;
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
    //根据需要调整
    app.measure_type = "qipao";
    app.cloth_type = "tieshen";
    app.gender = "male";

    #endif
    if(argc >= 2)
    {
        string filename(argv[1]);
        std::cout<<app.func(filename)<<endl;
        // app.measure_body(0,100);
        return 0;
    }
    else{
        // executing
        try { app.execute_ws(); }
        catch (const std::bad_alloc& /*e*/) { std::cout << "Bad alloc" << std::endl; }
        catch (const std::exception& /*e*/) { std::cout << "Exception" << std::endl; }
        std::cout<<"finished"<<std::endl;
        return 0;
    }
    
}
