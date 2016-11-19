#include "clone.h"
#include "poisson_solver.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>

bool findOverlap(cv::InputArray background,
                 cv::InputArray foreground,
                 int offsetX, int offsetY,
                 cv::Rect &rBackground,
                 cv::Rect &rForeground)
{
    cv::Mat bg = background.getMat();
    cv::Mat fg = foreground.getMat();

    
    rBackground = cv::Rect(0, 0, bg.cols, bg.rows) & 
                  cv::Rect(offsetX, offsetY, fg.cols, fg.rows);


    // Compensate for negative offsets. If offset < 0, offset in foreground is positive.
    rForeground = cv::Rect(std::max<int>(-offsetX, 0), 
                           std::max<int>(-offsetY, 0), 
                           rBackground.width, 
                           rBackground.height);

    
    return rForeground.area() > 0;
    
}

void computeMixedGradientVectorField(cv::InputArray background,
                                     cv::InputArray foreground,
                                     cv::OutputArray vx_,
                                     cv::OutputArray vy_)
{
    cv::Mat bg = background.getMat();
    cv::Mat fg = foreground.getMat();
    
    const int channels = bg.channels();
    
    vx_.create(bg.size(), CV_MAKETYPE(CV_32F, channels));
    vy_.create(bg.size(), CV_MAKETYPE(CV_32F, channels));
    
    cv::Mat vx = vx_.getMat();
    cv::Mat vy = vy_.getMat();
    
    cv::Mat kernelx = (cv::Mat_<float>(1, 3) << -0.5, 0, 0.5);
    cv::Mat kernely = (cv::Mat_<float>(3, 1) << -0.5, 0, 0.5);
    
    cv::Mat vxf, vyf, vxb, vyb;
    cv::filter2D(fg, vxf, CV_32F, kernelx, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(fg, vyf, CV_32F, kernely, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(bg, vxb, CV_32F, kernelx, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(bg, vyb, CV_32F, kernely, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    
    
    for(int id = 0; id <= (vx.rows * vx.cols * channels - channels); ++id)
    {
        const cv::Vec2f g[2] = {
            cv::Vec2f(vxf.ptr<float>()[id], vyf.ptr<float>()[id]),
            cv::Vec2f(vxb.ptr<float>()[id], vyb.ptr<float>()[id])
        };
        
        int which = (g[0].dot(g[0]) > g[1].dot(g[1])) ? 0 : 1;
        
        vx.ptr<float>()[id] = g[which][0];
        vy.ptr<float>()[id] = g[which][1];
    }
}
        
void computeWeightedGradientVectorField(cv::InputArray background,
                                        cv::InputArray foreground,
                                        cv::OutputArray vx,
                                        cv::OutputArray vy,
                                        float weightForeground)
{
    
    cv::Mat bg = background.getMat();
    cv::Mat fg = foreground.getMat();
    
    cv::Mat kernelx = (cv::Mat_<float>(1, 3) << -0.5, 0, 0.5);
    cv::Mat kernely = (cv::Mat_<float>(3, 1) << -0.5, 0, 0.5);
    
    cv::Mat vxf, vyf, vxb, vyb;
    cv::filter2D(fg, vxf, CV_32F, kernelx, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(fg, vyf, CV_32F, kernely, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(bg, vxb, CV_32F, kernelx, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(bg, vyb, CV_32F, kernely, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    
    cv::addWeighted(vxf, weightForeground, vxb, 1.f - weightForeground, 0, vx);
    cv::addWeighted(vyf, weightForeground, vyb, 1.f - weightForeground, 0, vy);
}
    
    
void seamlessClone(cv::InputArray background,
                 cv::InputArray foreground,
                 cv::InputArray foregroundMask,
                 int offsetX,
                 int offsetY,
                 cv::OutputArray destination,
                 CloneType type)
{
  
    // Copy original background as we only solve for the overlapping area of the translated foreground mask.
    background.getMat().copyTo(destination);
    
    // Find overlapping region. We will only perform on this region
    cv::Rect rbg, rfg;
    if (!findOverlap(background, foreground, offsetX, offsetY, rbg, rfg))
        return;
    
    cv::Mat fore, back;
    cv::Mat lap = (cv::Mat_<float>(3, 3) << 0.0, -1, 0.0, -1, 4, -1, 0.0, -1, 0.0);        
    cv::filter2D(foreground.getMat()(rfg), fore, CV_32F, lap, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(background.getMat()(rbg), back, CV_32F, lap, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    cv::Mat f = fore;

    cv::Mat boundaryMask(rfg.size(), CV_8UC1);      
    cv::threshold(foregroundMask.getMat()(rfg), boundaryMask, UNKNOWN, DIRICHLET_BD, cv::THRESH_BINARY_INV);
    cv::rectangle(boundaryMask, cv::Rect(0, 0, boundaryMask.cols, boundaryMask.rows), DIRICHLET_BD, 1);

    cv::Mat boundaryValues(rfg.size(), CV_MAKETYPE(CV_32F, background.channels()));
    background.getMat()(rbg).convertTo(boundaryValues, CV_32F);
    
    cv::Mat foreValues(rfg.size(), CV_MAKETYPE(CV_32F, foreground.channels()));
    foreground.getMat()(rfg).convertTo(foreValues, CV_32F);
    /*
    cv::Mat image = boundaryMask;
    int nr=image.rows;
    int nc=image.cols;
    if(image.isContinuous())
    {
        nr=1;
        nc=nc*image.rows*image.channels();
    }
    for(int i=0;i<nr;i++)
    {
       const uchar* inData=image.ptr<uchar>(i);
       for(int j=0;j<nc;j++)
       {
         std::cout<<int(*inData++)<<std::endl;

       }
    }
    */


   
    // Solve Poisson equation
    cv::Mat result;
    /*
    solvePoissonEquations(f,
                          boundaryMask,
                          boundaryValues,
                          result);
    */
    solvePoissonEquationsFast(foreValues,
                          boundaryMask,
                          boundaryValues,
                          result);
    
    // Copy result to destination image.
    result.convertTo(destination.getMat()(rbg), CV_8U);
    
}
    
    