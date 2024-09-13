#ifndef HEADER_SAVEPLY
#define HEADER_SAVEPLY
#include <string>
#include <opencv2/opencv.hpp>
#include <fstream>
using namespace std;
#define SAVE_PLY
//save points to ply
void saveToPly(std::vector<cv::Vec4f> &vertices, std::vector<cv::Vec4f> &normals,std::string name)
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

void saveToPlyColor(std::vector<cv::Vec4f> &vertices, std::vector<cv::Vec4f> &normals,std::string name, uint r, uint g, uint b)
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
    fscan<<"ply"<<endl<<"format ascii 1.0"<<endl<<"comment VCGLIB generated"<<endl;
    fscan<<"element vertex "<<num<<endl;
    fscan<<"property float x"<<endl<<"property float y"<<endl<<"property float z"<<endl;
    fscan<<"property float nx"<<endl<<"property float ny"<<endl<<"property float nz"<<endl;
    fscan<<"property uchar red"<<endl<<"property uchar blue"<<endl<<"property uchar green"<<endl;
    fscan<<"property uchar alpha"<<endl;
    fscan<<"end_header"<<endl;


    for(int i=0;i<vertices.size();i++){
        if(std::isnan(vertices[i][0]))
            continue;
        fscan<<vertices[i][0]<<" "<<vertices[i][1]<<" "<<vertices[i][2]<<" "<<normals[i][0]<<" "<<normals[i][1]<<" "<<normals[i][2]<<" "<<r<<" "<<g<<" "<<b<<" "<<255<<endl;
    }
    fscan.close();
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