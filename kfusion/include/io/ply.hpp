#ifndef HEADER_SAVEPLY
#define HEADER_SAVEPLY
#include <string>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
using namespace std;
#define SAVE_PLY
//save points to ply
void saveToPly(std::vector<cv::Vec4f> &vertices, std::vector<cv::Vec4f> &normals,std::string name)
{
    #ifndef SAVE_PLY
    return;
    #endif
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    for(const auto &v : vertices) {
        if(!std::isnan(v[0])) {
            pcl::PointXYZ point;
            point.x = v[0];
            point.y = v[1];
            point.z = v[2];
            cloud->points.push_back(point);
        }
    }
    
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = false;
    
    std::cout << "保存点云到文件" << std::endl;
    pcl::io::savePLYFile(name, *cloud);
}

void saveToPlyColor(std::vector<cv::Vec4f> &vertices, std::vector<cv::Vec4f> &normals,std::string name, uint r, uint g, uint b)
{
    #ifndef SAVE_PLY
    return;
    #endif
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

    for(size_t i = 0; i < vertices.size(); ++i) {
        if(!std::isnan(vertices[i][0])) {
            pcl::PointXYZRGBNormal point;
            point.x = vertices[i][0];
            point.y = vertices[i][1];
            point.z = vertices[i][2];
            point.normal_x = normals[i][0];
            point.normal_y = normals[i][1];
            point.normal_z = normals[i][2];
            point.r = r;
            point.g = g;
            point.b = b;
            cloud->points.push_back(point);
        }
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = false;

    std::cout << "正在保存点云到文件" << std::endl;
    pcl::io::savePLYFile(name, *cloud);
}

void saveToPly(std::vector<cv::Vec3f> &vertices, std::vector<cv::Vec3f> &normals,std::string name)
{
    #ifndef SAVE_PLY
    return;
    #endif
    int num = 0;
    for(auto &v:vertices)
    {
        if(!std::isnan(v[0]))
        {
            num++;
        }
    }
    cout<<"saving the point cloud to the file"<<endl;
    ofstream fscan(name);
    fscan<<"ply"<<endl<<"format ascii 1.0"<<endl<<"comment Created by Wang Junnan"<<endl;
    fscan<<"element vertex "<<num<<endl;
    fscan<<"property float x"<<endl<<"property float y"<<endl<<"property float z"<<endl;
    fscan<<"end_header"<<endl;

    for(int i=0;i<vertices.size();i++){
        if(std::isnan(vertices[i][0]))
            continue;
        fscan<<vertices[i][0]<<" "<<vertices[i][1]<<" "<<vertices[i][2]<<endl;
    }
    fscan.close();
}

void saveToPlyColor(std::vector<cv::Vec3f> &vertices, std::vector<cv::Vec3f> &normals,std::string name, uint r, uint g, uint b)
{
    #ifndef SAVE_PLY
    return;
    #endif
    int num = 0;
    for(auto &v:vertices)
    {
        if(!std::isnan(v[0]))
        {
            num++;
        }
    }
    cout<<"saving the point cloud to the file"<<endl;
    ofstream fscan(name);
    fscan<<"ply"<<endl<<"format ascii 1.0"<<endl<<"comment Created by Wang Junnan"<<endl;
    fscan<<"element vertex "<<num<<endl;
    fscan<<"property float x"<<endl<<"property float y"<<endl<<"property float z"<<endl;
    fscan<<"property uint8 red"<<endl<<"property uint8 green"<<endl<<"property uint8 blue"<<endl;
    fscan<<"end_header"<<endl;

    for(int i=0;i<vertices.size();i++){
        if(std::isnan(vertices[i][0]))
            continue;
        fscan<<vertices[i][0]<<" "<<vertices[i][1]<<" "<<vertices[i][2]<<" "<<r<<" "<<g<<" "<<b<<endl;
    }
    fscan.close();
}
/// @brief  for debug, to change the xyz of the object
/// @param name 
void changeXYZ_ply(string name)
{
    ifstream fobj(name);
    string line;
    float x, y, z;
    float nx, ny, nz;
    unsigned char r, g ,b;
    bool flag_start = false;
    int vertex_num = 0;
    int idx = 0;
    if (fobj.is_open())
    {
        while (!fobj.eof())
        {
            std::getline(fobj, line);
            std::istringstream ssline(line);
            if(flag_start)
            {
                ssline >> x;
                ssline >> y;
                ssline >> z;
                ssline >> nx;
                ssline >> ny;
                ssline >> nz;
                ssline >> r;
                ssline >> g;
                ssline >> b;
                cout<<x<<","<<y<<","<<z<<endl;
                idx ++;
                if(idx >= vertex_num)
                {
                    break;
                }
            }
            else{
                
                if(line == "end_header")
                {
                    flag_start = true;
                }
                if(line.find("element vertex")!=string::npos)
                {
                    vertex_num = std::stoi(line.substr(14));
                }
            }
        }
        fobj.close();
        
    }
    else{
        cout<<"ERROR, file "<<name<<" open failed, CHECK"<<endl;
    }
    
}
#endif