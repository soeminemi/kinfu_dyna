#ifndef HEADER_SAVEPLY
#define HEADER_SAVEPLY
#include <string>
#include <opencv2/opencv.hpp>
#include <fstream>
using namespace std;
//save points to ply
void saveToPly(std::vector<cv::Vec4f> &vertices, std::string name)
{
    cout<<"saving the point cloud to the file"<<endl;
    ofstream fscan(name);
    fscan<<"ply"<<endl<<"format ascii 1.0"<<endl<<"comment Created by Wang Junnan"<<endl;
    fscan<<"element vertex "<<vertices.size()<<endl;
    fscan<<"property float x"<<endl<<"property float y"<<endl<<"property float z"<<endl;
    fscan<<"end_header"<<endl;

    for(int i=0;i<vertices.size();i++){
        fscan<<vertices[i][0]<<" "<<vertices[i][1]<<" "<<vertices[i][2]<<endl;
    }
    fscan.close();
}
#endif