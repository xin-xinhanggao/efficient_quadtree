#ifndef POISSON_CLONE_H
#define POISSON_CLONE_H

#include "constant.h"
#include <opencv2/core/core.hpp>
    
/**
 
 Determine the area of overlap.
 
 Computes the areas of overlap for background and foreground when foreground
 is layed over background given a translational offset.
 
 */
bool findOverlap(cv::InputArray background,
                 cv::InputArray foreground,
                 int offsetX, int offsetY,
                 cv::Rect &rBackground,
                 cv::Rect &rForeground);

/** 
 
 Compute Poisson guidance vector field by mixing gradients from background and foreground.
 
 */
void computeMixedGradientVectorField(cv::InputArray background,
                                     cv::InputArray foreground,
                                     cv::OutputArray vx,
                                     cv::OutputArray vy);

/**
 
 Compute Poisson guidance vector field by averaging background and foreground gradients.
 
 */
void computeWeightedGradientVectorField(cv::InputArray background,
                                        cv::InputArray foreground,
                                        cv::OutputArray vx,
                                        cv::OutputArray vy,
                                        float weightForeground);

/** 
 
 Solve multi-channel Poisson equations.
 
 */
void solvePoissonEquations(cv::InputArray background,
                           cv::InputArray foreground,
                           cv::InputArray foregroundMask,
                           cv::InputArray vx,
                           cv::InputArray vy,
                           cv::OutputArray destination);
    
    
enum CloneType {
    CLONE_FOREGROUND_GRADIENTS,
    CLONE_AVERAGED_GRADIENTS,
    CLONE_MIXED_GRADIENTS
};


void seamlessClone(cv::InputArray background,
                   cv::InputArray foreground,
                   cv::InputArray foregroundMask,
                   int offsetX,
                   int offsetY,
                   cv::OutputArray destination,
                   CloneType type);

void seamlessCloneNaive(cv::InputArray background,
                   cv::InputArray foreground,
                   cv::InputArray foregroundMask,
                   int offsetX,
                   int offsetY,
                   cv::OutputArray destination,
                   CloneType type);

#endif