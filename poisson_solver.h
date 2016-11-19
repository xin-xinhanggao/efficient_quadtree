#ifndef POISSON_SOLVER_H
#define POISSON_SOLVER_H

#include "constant.h"
#include <opencv2/core/core.hpp>


/**        
Solve multi-channel Poisson equations on rectangular domain.
*/
void solvePoissonEquations(
    cv::InputArray f,
    cv::InputArray bdMask,
    cv::InputArray bdValues,
    cv::OutputArray result);

void solvePoissonEquationsFast(
    cv::InputArray f,
    cv::InputArray bdMask,
    cv::InputArray bdValues,
    cv::OutputArray result);

#endif