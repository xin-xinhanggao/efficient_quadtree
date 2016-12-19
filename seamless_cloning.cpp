#include "clone.h"
#include <opencv2/opencv.hpp>

/**
 
 Naive image cloning by just copying the values from foreground over background
 
 */
void naiveClone(cv::InputArray background_,
                cv::InputArray foreground_,
                cv::InputArray foregroundMask_,
                int offsetX, int offsetY,
                cv::OutputArray destination_)
{
    cv::Mat bg = background_.getMat();
    cv::Mat fg = foreground_.getMat();
    cv::Mat fgm = foregroundMask_.getMat();
    
    destination_.create(bg.size(), bg.type());
    cv::Mat dst = destination_.getMat();
    
    cv::Rect overlapAreaBg, overlapAreaFg;
    findOverlap(background_, foreground_, offsetX, offsetY, overlapAreaBg, overlapAreaFg);
    
    bg.copyTo(dst);
    fg(overlapAreaFg).copyTo(dst(overlapAreaBg), fgm(overlapAreaFg));
}

/**
 
 Main entry point.
 
 */
int main(int argc, char **argv)
{
    if (argc != 6) {
        std::cerr << argv[0] << " background foreground mask offsetx offsety" << std::endl;
        return -1;
    }
    
    cv::Mat background = cv::imread(argv[1]);
    cv::Mat foreground = cv::imread(argv[2]);
    cv::Mat mask = cv::imread(argv[3], CV_LOAD_IMAGE_GRAYSCALE);
    int offsetx = atoi(argv[4]);
    int offsety = atoi(argv[5]);
    
    
    cv::Mat result;
    
    naiveClone(background, foreground, mask, offsetx, offsety, result);
    cv::imshow("Naive", result);
    cv::imwrite("naive.png", result);

    seamlessClone(background, foreground, mask, offsetx, offsety, result, CLONE_AVERAGED_GRADIENTS);
    cv::imshow("source Gradients", result);
    cv::imwrite("source-gradients.png", result);
    
    cv::waitKey();
    
    return 0;
}




