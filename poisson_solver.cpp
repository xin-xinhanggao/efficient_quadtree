#include "poisson_solver.h"
#include <opencv2/opencv.hpp>
#pragma warning (push)
#pragma warning (disable: 4244)
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <ctime>
#include <cstdlib>
#pragma warning (pop)
#include <bitset>
#include <set>
      

class Node
{
public:
    Node(int x,int y,int size)
    {
        this->x = x;
        this->y = y;
        this->size = size;
    }
    int size;
    int x;
    int y;
    friend inline bool operator<(const Node &a, const Node &b);
};

inline bool operator<(const Node &a, const Node &b)
{
    return (a.size > b.size) || (a.size == b.size && a.x > b.x) || (a.size == b.size && a.x == b.x && a.y > b.y);
}

bool isSameSize(cv::Size a, cv::Size b) {
    return a.width == b.width && a.height == b.height;
}

/* Make matrix memory continuous. */
cv::Mat makeContinuous(cv::Mat m) {       
    if (!m.isContinuous()) {
        m = m.clone();
    }        
    return m;
}

/* Build a one dimensional index lookup for element in mask. */
cv::Mat buildPixelToIndexLookup(cv::InputArray mask, int &npixel)
{
    cv::Mat_<uchar> m = makeContinuous(mask.getMat());

    cv::Mat_<int> pixelToIndex(mask.size());
    npixel = 0;
    
    int *pixelToIndexPtr = pixelToIndex.ptr<int>();
    const uchar *maskPtr = m.ptr<uchar>();

    for (int id = 0; id < (m.rows * m.cols); ++id) {
        pixelToIndexPtr[id] = (maskPtr[id] == DIRICHLET_BD) ? -1 : npixel++;
    }

    return pixelToIndex;
}

cv::Mat buildPixelToIndexLookup(cv::InputArray mask, cv::InputArray quadtree, int &npixel)
{
    cv::Mat_<uchar> m = makeContinuous(mask.getMat());
    cv::Mat quad = makeContinuous(quadtree.getMat());

    cv::Mat_<int> pixelToIndex(mask.size());
    npixel = 0;

    int *pixelToIndexPtr = pixelToIndex.ptr<int>();
    const uchar *maskPtr = m.ptr<uchar>();
    const float *quadPtr = quad.ptr<float>();

    for (int id = 0; id < (m.rows * m.cols); ++id) {
        
        if(maskPtr[id] == DIRICHLET_BD)
            pixelToIndexPtr[id] = -1;
        else if(quadPtr[id] > 0)
        {
            pixelToIndexPtr[id] = npixel++;
        }
        else
        {
            pixelToIndexPtr[id] = -1;
        }
    }

    return pixelToIndex;
}


static int nearest_powerof2(int nrows)
{
    unsigned i;
    nrows -= 1;
    for(i=1; i<sizeof(nrows) * 8; i <<= 1)
        nrows = nrows | (nrows >> i);
    nrows += 1;
 
    return nrows;
}

int quadfind(cv::InputArray quadtree, int mode, int row, int col, int height, int width)
{
    if(row > height || col > width)
        return -1;

    if(mode == 0) //right 
    {
        for(int i = col + 1; i < width; i++)
            if(quadtree.getMat().at<float>(row, i) > 0)
                return i;
        return -1;
    }

    if(mode == 1) //left
    {
        for(int i = col - 1;i > -1; i--)
            if(quadtree.getMat().at<float>(row, i) > 0)
                return i;
        return -1;
    }

    if(mode == 2) //bottom
    {
        for(int i = row + 1;i < height;i++)
            if(quadtree.getMat().at<float>(i, col) > 0)
                return i;
        return -1;
    }

    if(mode == 3) //up
    {
        for(int i = row - 1;i > -1; i--)
            if(quadtree.getMat().at<float>(i, col) > 0)
                return i;
        return -1;
    }

    return -1;
}

void initquad(cv::InputArray f_,
    cv::InputArray bdMask_,
    cv::InputArray bdValues_,
    cv::OutputArray seaminresult,
    cv::OutputArray seamoutresult,
    cv::OutputArray seam)
{

    cv::Mat seamin,seamout;
    cv::Mat bdseamout;
    cv::Mat seamcal = (cv::Mat_<float>(3, 3) << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);        
    cv::filter2D(bdMask_, seam, CV_32F, seamcal);
    cv::threshold(seam, seamout, 0, 1, cv::THRESH_BINARY);
    cv::threshold(seam, seamin, 8, 1, cv::THRESH_BINARY_INV);
    seamin.copyTo(seaminresult, bdMask_);  
    cv::threshold(bdMask_, bdseamout, 0, 1, cv::THRESH_BINARY_INV);

    seamout.copyTo(seamoutresult, bdseamout);
    
    seam.getMat() = seaminresult.getMat() + seamoutresult.getMat();
    //seamoutresult.getMat().copyTo(seam.getMat());

    
    cv::Mat ones(seam.getMat().rows, seam.getMat().cols, CV_32F);
    ones.setTo(1);
    cv::Rect rect1(0,ones.rows - 1,ones.cols,1);
    cv::Rect rect2(ones.cols - 1,0,1,ones.rows);
    ones(rect1).copyTo(seam.getMat()(rect1));
    ones(rect2).copyTo(seam.getMat()(rect2));

}


void solvenode(std::set<Node> &tree, cv::InputArray seam, cv::OutputArray quadtree, int x, int y, int dim, int height, int width)
{
    if(dim == 1) //means leaf node
    {
        quadtree.getMat().at<float>(x,y) = 1;
        if(x < height && y < width)
            tree.insert(Node(x,y,dim));
        return;
    }
    cv::Rect quadrect(y,x,dim,dim);
    if(countNonZero(seam.getMat()(quadrect)) == 0) // means all 0
    {
        if(x < height && y < width)
            tree.insert(Node(x,y,dim));
        quadtree.getMat().at<float>(x,y) = dim;
        return;
    }
    else
    {
        solvenode(tree, seam, quadtree, x, y, dim / 2, height, width); // left-up
        solvenode(tree, seam, quadtree, x, y + dim / 2, dim / 2, height, width); // right-up
        solvenode(tree, seam, quadtree, x + dim / 2, y, dim / 2, height, width); // left-down
        solvenode(tree, seam, quadtree, x + dim / 2, y + dim / 2, dim / 2, height, width); //right-down
    }
}

void interpolate(cv::InputArray quadtree, std::set<Node> &tree, cv::OutputArray residual)
{
    
    int height = quadtree.getMat().rows;
    int width = quadtree.getMat().cols;
    cv::Mat r = residual.getMat();
    
    for(std::set<Node>::iterator it = tree.begin(); it != tree.end(); it++)
    {
        int bottomNNZ = 0, rightNNZ = 0;
        int dim = it->size;
        int x = it->x;
        int y = it->y;

        if(dim < 2)
            continue;

        if(x + dim < height)
        {
            cv::Rect bottom(y, x + dim, dim, 1);
            bottomNNZ = countNonZero(quadtree.getMat()(bottom));
        }

        if(y + dim < width)
        {
            cv::Rect right(y + dim, x, 1, dim);
            rightNNZ = countNonZero(quadtree.getMat()(right));
        }

        if(bottomNNZ == 2)
        {
            for(int channel = 0; channel < 3; channel++)
                r.ptr<float>(x + dim)[3 * (y + dim / 2) + channel] = (r.ptr<float>(x + dim)[3 * (y) + channel] + r.ptr<float>(x + dim)[3 * (y + dim) + channel]) / 2;
        }

        if(rightNNZ == 2)
        {
            for(int channel = 0; channel < 3; channel++)
                r.ptr<float>(x + dim / 2)[3 * (y + dim) + channel] = (r.ptr<float>(x)[3 * (y + dim) + channel] + r.ptr<float>(x + dim)[3 * (y + dim) + channel]) / 2;
        }
    }

    
    for(std::set<Node>::iterator it = tree.begin(); it != tree.end(); it++)
    {
        int dim = it->size;
        int x = it->x;
        int y = it->y;

        if(dim < 2)
            continue;

        for(int index = 1; index < dim; index++)
        {
            float blendingFactor = index * 1.0 / dim;
            for(int channel = 0; channel < 3; channel++)
                r.ptr<float>(x)[3 * (y + index) + channel] = (1 - blendingFactor) * r.ptr<float>(x)[3 * (y) + channel] + blendingFactor * r.ptr<float>(x)[3 * (y + dim) + channel];
        }

        for(int index = 1; index < dim; index++)
        {
            float blendingFactor = index * 1.0 / dim;
            for(int channel = 0; channel < 3; channel++)
                r.ptr<float>(x + index)[3 * (y) + channel] = (1 - blendingFactor) * r.ptr<float>(x)[3 * (y) + channel] + blendingFactor * r.ptr<float>(x + dim)[3 * (y) + channel];
        }
    }
    
    for(std::set<Node>::iterator it = tree.begin(); it != tree.end(); it++)
    {
        int dim = it->size;
        int x = it->x;
        int y = it->y;

        if(dim < 2)
            continue;

        for(int xindex = 1; xindex < dim; xindex++)
        for(int yindex = 1; yindex < dim; yindex++)
        {
            float blendingFactor = xindex * 1.0 / dim;
            for(int channel = 0; channel < 3; channel++)
                r.ptr<float>(x + xindex)[3 * (y + yindex) + channel] = (1 - blendingFactor) * r.ptr<float>(x)[3 * (y + yindex) + channel] + blendingFactor * r.ptr<float>(x + dim)[3 * (y + yindex) + channel];
        }
    }

}

void refine(std::set<Node> &leafnode, cv::OutputArray quadtree, int height, int width)
{
    int changeflag = 1;
    while(changeflag != 0)
    {
        changeflag = 0;
        for(std::set<Node>::iterator it = leafnode.begin(); it != leafnode.end(); it++)
        {
            int topNNZ = 0;
            int bottomNNZ = 0;
            int leftNNZ = 0;
            int rightNNZ = 0;

            int x = it->x;
            int y = it->y;
            int dim = it->size;
            if(x - dim / 2 >= 0)
            {
                cv::Rect top(y, x - dim / 2, dim, dim / 2);
                topNNZ = countNonZero(quadtree.getMat()(top));
            }

            if(x + dim < height)
            {
                cv::Rect bottom(y, x + dim, dim, 1);
                bottomNNZ = countNonZero(quadtree.getMat()(bottom));
            }

            if(y - dim / 2 >= 0)
            {
                cv::Rect left(y - dim / 2, x, dim / 2,dim);
                leftNNZ = countNonZero(quadtree.getMat()(left));
            }

            if(y + dim < width)
            {
                cv::Rect right(y + dim, x, 1, dim);
                rightNNZ = countNonZero(quadtree.getMat()(right));
            }

            if(topNNZ > 2 || bottomNNZ > 2 || leftNNZ > 2|| rightNNZ > 2)
            {
                quadtree.getMat().at<float>(x,y) = dim / 2;
                quadtree.getMat().at<float>(x + dim / 2,y) = dim / 2;
                quadtree.getMat().at<float>(x,y + dim / 2) = dim / 2;
                quadtree.getMat().at<float>(x + dim / 2,y + dim / 2) = dim / 2;
                leafnode.erase(Node(x,y,dim));
                leafnode.insert(Node(x,y,dim / 2));
                leafnode.insert(Node(x + dim / 2,y,dim / 2));
                leafnode.insert(Node(x,y + dim / 2,dim / 2));
                leafnode.insert(Node(x + dim / 2,y + dim / 2,dim / 2));
                changeflag++;
            }
        }
    }
}


void solvePoissonEquationsFast( //f means source value 
                                //bdValues_ means background value
                                //bdMask_ mans mask region the image
    cv::InputArray f_,
    cv::InputArray bdMask_,
    cv::InputArray bdValues_,
    cv::OutputArray result_)
{
    CV_Assert(
        !f_.empty() &&
        isSameSize(f_.size(), bdMask_.size()) &&
        isSameSize(f_.size(), bdValues_.size())
    );

    CV_Assert(
        f_.depth() == CV_32F &&
        bdMask_.depth() == CV_8U &&
        bdValues_.depth() == CV_32F &&
        f_.channels() == bdValues_.channels() &&
        bdMask_.channels() == 1);

    cv::Mat seaminresult, seamoutresult, seam;    
    initquad(f_, bdMask_, bdValues_, seaminresult, seamoutresult, seam);
    
    int pow2_size = std::max(nearest_powerof2(seam.cols), nearest_powerof2(seam.rows));
    cv::Mat quadtree(pow2_size, pow2_size, CV_32F);
    quadtree.setTo(0);
    cv::Mat quadseam(pow2_size, pow2_size, CV_32F);
    cv::Rect quadrect(0, 0, seam.cols, seam.rows);
    seam.copyTo(quadseam(quadrect));

    std::set<Node> leafnode; // the set of leaf node
    solvenode(leafnode, quadseam, quadtree, 0, 0, pow2_size, seam.rows, seam.cols);
    refine(leafnode, quadtree, seam.rows, seam.cols);

    quadtree = quadtree(quadrect);

    cv::Mat quadMat(quadtree.rows, quadtree.cols, CV_32F);
    srand((unsigned)time(NULL)); 
    for(std::set<Node>::iterator it = leafnode.begin(); it != leafnode.end(); it++)
    {
        float color = rand() / double(RAND_MAX);
        for(int x = 0; x < it->size; x++)
        for(int y = 0; y < it->size; y++)
        {
            quadMat.at<float>(it->x + x,it->y + y) = color;
        }
    }
    //cv::imwrite("quadMat.png", quadMat * 255.);

    cv::Mat f = makeContinuous(f_.getMat());
    cv::Mat_<uchar> bm = makeContinuous(bdMask_.getMat());
    cv::Mat bv = makeContinuous(bdValues_.getMat());
    cv::Mat composed;

    result_.create(f.size(), f.type());
    composed.create(f.size(), f.type());


    f.copyTo(composed); // through this the composed matrix comes to itself
    bv.copyTo(composed, bm == DIRICHLET_BD);

    cv::Mat r = result_.getMat();
    
    int nUnknowns = 0;
    cv::Mat_<int> unknownIdx = buildPixelToIndexLookup(bm, quadtree, nUnknowns);

    if (nUnknowns == 0) {
        // No unknowns left, we're done
        return;
    } else if (nUnknowns == f.size().area()) {
        // All unknowns, will not lead to a unique solution
        // TODO emit warning
    }

    const int channels = f.channels();
    std::vector< Eigen::Triplet<float> > lhsTriplets;
    lhsTriplets.reserve(nUnknowns * 9);

    Eigen::MatrixXf rhs(nUnknowns, channels);
    Eigen::MatrixXf reserve(nUnknowns, channels);
    rhs.setZero();
    reserve.setZero();

    for (int row = 0; row < f.rows; ++row) 
    {
        for (int col = 0; col < f.cols; ++col) 
        {

            const cv::Point p(col, row);
            const int pid = unknownIdx(p);

            if (pid == -1) {
                // Current pixel is not an unknown, skip
                continue;
            }

            int lab = 4;
            int dim = quadtree.at<float>(row, col);  


            int right = quadfind(quadtree, 0, row, col, seam.rows, seam.cols);
            if(right != -1)
            {
                if(right - col > dim) // the simple condition is equivalent
                {
                    int top = unknownIdx(cv::Point(col + dim, row - dim));
                    int bottom = unknownIdx(cv::Point(col + dim, row + dim));

                    if(top != -1)
                        lhsTriplets.push_back(Eigen::Triplet<float>(pid, top, -0.5));

                    if(bottom != -1)
                        lhsTriplets.push_back(Eigen::Triplet<float>(pid, bottom, -0.5));

                    reserve.row(pid) -= Eigen::Map<Eigen::VectorXf>(composed.ptr<float>(row, col), channels) - 0.5 * Eigen::Map<Eigen::VectorXf>(composed.ptr<float>(row - dim, col + dim), channels)
                     - 0.5 * Eigen::Map<Eigen::VectorXf>(composed.ptr<float>(row + dim, col + dim), channels);

                    reserve.row(pid) += Eigen::Map<Eigen::VectorXf>(f.ptr<float>(row, col), channels) - 0.5 * Eigen::Map<Eigen::VectorXf>(f.ptr<float>(row - dim, col + dim), channels)
                     - 0.5 * Eigen::Map<Eigen::VectorXf>(f.ptr<float>(row + dim, col + dim), channels);
                }
                else
                {
                    int next = unknownIdx(cv::Point(right, row));
                    if(next != -1) 
                        lhsTriplets.push_back(Eigen::Triplet<float>(pid, next, -1));

                    reserve.row(pid) -= Eigen::Map<Eigen::VectorXf>(composed.ptr<float>(row, col), channels) -  Eigen::Map<Eigen::VectorXf>(composed.ptr<float>(row, right), channels);
                    reserve.row(pid) += Eigen::Map<Eigen::VectorXf>(f.ptr<float>(row, col), channels) -  Eigen::Map<Eigen::VectorXf>(f.ptr<float>(row, right), channels);
                }
            }
            else
                lab--;

            int left = quadfind(quadtree, 1, row, col, seam.rows, seam.cols);
            if(left != -1)
            {
                if(col - 2 * dim > -1 && row - dim > -1 && (row + dim - 1) < f.rows && (quadtree.at<float>(row - dim, col - 2 * dim) > (2 * dim - 1) && quadtree.at<float>(row - dim, col - 2 * dim) < (2 * dim + 1)))
                {
                        int top = unknownIdx(cv::Point(col, row - dim));
                        int bottom = unknownIdx(cv::Point(col, row + dim));

                        if(top != -1)
                            lhsTriplets.push_back(Eigen::Triplet<float>(pid, top, -0.5));

                        if(bottom != -1)
                            lhsTriplets.push_back(Eigen::Triplet<float>(pid, bottom, -0.5));

                        reserve.row(pid) -= Eigen::Map<Eigen::VectorXf>(composed.ptr<float>(row, col), channels) - 0.5 * Eigen::Map<Eigen::VectorXf>(composed.ptr<float>(row - dim, col), channels)
                     - 0.5 * Eigen::Map<Eigen::VectorXf>(composed.ptr<float>(row + dim, col), channels);

                        reserve.row(pid) += Eigen::Map<Eigen::VectorXf>(f.ptr<float>(row, col), channels) - 0.5 * Eigen::Map<Eigen::VectorXf>(f.ptr<float>(row - dim, col), channels)
                     - 0.5 * Eigen::Map<Eigen::VectorXf>(f.ptr<float>(row + dim, col), channels);
                }
                else
                {
                    int next = unknownIdx(cv::Point(left, row));
                    if(next != -1)
                        lhsTriplets.push_back(Eigen::Triplet<float>(pid, next, -1));

                    reserve.row(pid) -= Eigen::Map<Eigen::VectorXf>(composed.ptr<float>(row, col), channels) -  Eigen::Map<Eigen::VectorXf>(composed.ptr<float>(row, left), channels);
                    reserve.row(pid) += Eigen::Map<Eigen::VectorXf>(f.ptr<float>(row, col), channels) -  Eigen::Map<Eigen::VectorXf>(f.ptr<float>(row, left), channels);
                }
            }
            else
                lab--;

            int down = quadfind(quadtree, 2, row, col, seam.rows, seam.cols);
            if(down != -1)
            {
                if(down - row > dim) //T junction
                {
                    int leftblock = unknownIdx(cv::Point(col - dim, row + dim));
                    int rightblock = unknownIdx(cv::Point(col + dim, row + dim));

                    if(leftblock != -1)
                        lhsTriplets.push_back(Eigen::Triplet<float>(pid, leftblock, -0.5));

                    if(rightblock != -1)
                        lhsTriplets.push_back(Eigen::Triplet<float>(pid, rightblock, -0.5));

                    reserve.row(pid) -= Eigen::Map<Eigen::VectorXf>(composed.ptr<float>(row, col), channels) - 0.5 * Eigen::Map<Eigen::VectorXf>(composed.ptr<float>(row + dim, col - dim), channels)
                     - 0.5 * Eigen::Map<Eigen::VectorXf>(composed.ptr<float>(row + dim, col + dim), channels);

                    reserve.row(pid) += Eigen::Map<Eigen::VectorXf>(f.ptr<float>(row, col), channels) - 0.5 * Eigen::Map<Eigen::VectorXf>(f.ptr<float>(row + dim, col - dim), channels)
                     - 0.5 * Eigen::Map<Eigen::VectorXf>(f.ptr<float>(row + dim, col + dim), channels);
                }
                else
                {
                    int next = unknownIdx(cv::Point(col, down));
                    if(next != -1)
                        lhsTriplets.push_back(Eigen::Triplet<float>(pid, next, -1));

                    reserve.row(pid) -= Eigen::Map<Eigen::VectorXf>(composed.ptr<float>(row, col), channels) - Eigen::Map<Eigen::VectorXf>(composed.ptr<float>(down, col), channels);
                    reserve.row(pid) += Eigen::Map<Eigen::VectorXf>(f.ptr<float>(row, col), channels) - Eigen::Map<Eigen::VectorXf>(f.ptr<float>(down, col), channels);
                }
            }
            else
                lab--;

            int up = quadfind(quadtree, 3, row, col, seam.rows, seam.cols);
            if(up != -1)
            {
                if(col - dim > -1 && row - 2 * dim > -1 && col + dim - 1< f.cols && quadtree.at<float>(row - 2 * dim, col - dim) > (2 * dim - 1) && quadtree.at<float>(row - 2 * dim, col - dim) < (2 * dim + 1))
                {
                    int leftblock = unknownIdx(cv::Point(col - dim, row));
                    int rightblock = unknownIdx(cv::Point(col + dim, row));

                    if(leftblock != -1)
                        lhsTriplets.push_back(Eigen::Triplet<float>(pid, leftblock, -0.5));

                    if(rightblock != -1)
                        lhsTriplets.push_back(Eigen::Triplet<float>(pid, rightblock, -0.5));

                    reserve.row(pid) -= Eigen::Map<Eigen::VectorXf>(composed.ptr<float>(row, col), channels) - 0.5 * Eigen::Map<Eigen::VectorXf>(composed.ptr<float>(row, col - dim), channels)
                     - 0.5 * Eigen::Map<Eigen::VectorXf>(composed.ptr<float>(row, col + dim), channels);

                    reserve.row(pid) += Eigen::Map<Eigen::VectorXf>(f.ptr<float>(row, col), channels) - 0.5 * Eigen::Map<Eigen::VectorXf>(f.ptr<float>(row, col - dim), channels)
                     - 0.5 * Eigen::Map<Eigen::VectorXf>(f.ptr<float>(row, col + dim), channels);
                }
                else
                {
                    int next = unknownIdx(cv::Point(col, up));
                    if(next != -1)
                        lhsTriplets.push_back(Eigen::Triplet<float>(pid, next, -1));

                    reserve.row(pid) -= Eigen::Map<Eigen::VectorXf>(composed.ptr<float>(row, col), channels) - Eigen::Map<Eigen::VectorXf>(composed.ptr<float>(up, col), channels);
                    reserve.row(pid) += Eigen::Map<Eigen::VectorXf>(f.ptr<float>(row, col), channels) - Eigen::Map<Eigen::VectorXf>(f.ptr<float>(up, col), channels);
                }
            }
            else
                lab--;

            lhsTriplets.push_back(Eigen::Triplet<float>(pid, pid, lab));
            if(seamoutresult.at<float>(row, col) > 0)
                rhs.row(pid) += reserve.row(pid);
        }
    }

    std::cout<<"complete get matrix A"<<std::endl;
    std::cout<<"unknowns "<<nUnknowns<<std::endl;
    Eigen::SparseMatrix<float> A(nUnknowns, nUnknowns);
    A.setFromTriplets(lhsTriplets.begin(), lhsTriplets.end());

    Eigen::SparseLU< Eigen::SparseMatrix<float> > solver;
    solver.analyzePattern(A);
    solver.factorize(A);

    Eigen::MatrixXf result(nUnknowns, channels);

    for (int c = 0; c < channels; ++c)
        result.col(c) = solver.solve(rhs.col(c));
    std::cout<<"complete solve system"<<std::endl;

    cv::Mat residual;
    residual.create(f.size(), f.type());

    for(std::set<Node>::iterator it = leafnode.begin(); it != leafnode.end(); it++)
    {
        const cv::Point p(it->y, it->x);
        const int pid = unknownIdx(p);
        if(pid > -1)
        {
            Eigen::Map<Eigen::VectorXf>(residual.ptr<float>(p.y, p.x), channels) = result.row(pid);
        }
    }

    interpolate(quadtree, leafnode, residual);

    cv::Mat clearresidual = cv::Mat::zeros(f.size(), f.type());
    residual.copyTo(clearresidual, bm == 0);

    r = composed + clearresidual;
}

void solvePoissonEquations(
    cv::InputArray f_,
    cv::InputArray bdMask_,
    cv::InputArray bdValues_,
    cv::OutputArray result_)
{
        // Input validation

    CV_Assert(
        !f_.empty() &&
        isSameSize(f_.size(), bdMask_.size()) &&
        isSameSize(f_.size(), bdValues_.size())
    );

    CV_Assert(
        f_.depth() == CV_32F &&
        bdMask_.depth() == CV_8U &&
        bdValues_.depth() == CV_32F &&
        f_.channels() == bdValues_.channels() &&
        bdMask_.channels() == 1);

    // We assume continuous memory on input
    cv::Mat f = makeContinuous(f_.getMat());
    cv::Mat_<uchar> bm = makeContinuous(bdMask_.getMat());
    cv::Mat bv = makeContinuous(bdValues_.getMat());

    // Allocate output
    result_.create(f.size(), f.type());
    cv::Mat r = result_.getMat();
    bv.copyTo(r, bm == DIRICHLET_BD);

    // The number of unknowns correspond to the number of pixels on the rectangular region 
    // that don't have a Dirichlet boundary condition.
    int nUnknowns = 0;
    cv::Mat_<int> unknownIdx = buildPixelToIndexLookup(bm, nUnknowns);

    if (nUnknowns == 0) {
        // No unknowns left, we're done
        return;
    } else if (nUnknowns == f.size().area()) {
        // All unknowns, will not lead to a unique solution
        // TODO emit warning
    }

    const cv::Rect bounds(0, 0, f.cols, f.rows);

    // Directional indices
    const int center = 0;
    const int north = 1;
    const int east = 2;
    const int south = 3;
    const int west = 4;

    // Neighbor offsets in all directions
    const int offsets[5][2] = { { 0, 0 }, { 0, -1 }, { 1, 0 }, { 0, 1 }, { -1, 0 } };
    
    // Directional opposite
    const int opposite[5] = { center, south, west, north, east };
    const int channels = f.channels();
    
    std::vector< Eigen::Triplet<float> > lhsTriplets;
    lhsTriplets.reserve(nUnknowns * 5);

    Eigen::MatrixXf rhs(nUnknowns, channels);
    rhs.setZero();
    
    // Loop over domain once. The coefficient matrix A is the same for all
    // channels, the right hand side is channel dependent.

    for (int y = 0; y < f.rows; ++y) 
    {
        for (int x = 0; x < r.cols; ++x) 
        {

            const cv::Point p(x, y);
            const int pid = unknownIdx(p);

            if (pid == -1) {
                // Current pixel is not an unknown, skip
                continue;
            }

            // Start coefficients of left hand side. Based on discrete Laplacian with central difference.
            float lhs[] = { -4.f, 1.f, 1.f, 1.f, 1.f };
            
            
            for (int n = 1; n < 5; ++n) 
            {
                const cv::Point q(x + offsets[n][0], y + offsets[n][1]);
                
                const bool hasNeighbor = bounds.contains(q);
                const bool isNeighborDirichlet = hasNeighbor && (bm(q) == DIRICHLET_BD);
                
                if (!hasNeighbor) 
                {
                    lhs[center] += lhs[n];
                    lhs[n] = 0.f;
                } 
                else if (isNeighborDirichlet) 
                {
                    
                    // Implementation note:
                    //
                    // Dirichlet boundary conditions (DB) turn neighbor unknowns into knowns (data) and
                    // are therefore moved to the right hand side. Alternatively, we could add more
                    // equations for these pixels setting the lhs 1 and rhs to the Dirichlet value, but
                    // that would unnecessarily blow up the equation system.
                    
                    rhs.row(pid) -= lhs[n] * Eigen::Map<Eigen::VectorXf>(bv.ptr<float>(q.y, q.x), channels);
                    lhs[n] = 0.f;
                }
            }


            // Add f to rhs.
            rhs.row(pid) += Eigen::Map<Eigen::VectorXf>(f.ptr<float>(p.y, p.x), channels);

            // Build triplets for row              
            for (int n = 0; n < 5; ++n) {
                if (lhs[n] != 0.f) {
                    const cv::Point q(x + offsets[n][0], y + offsets[n][1]);
                    lhsTriplets.push_back(Eigen::Triplet<float>(pid, unknownIdx(q), lhs[n]));
                }
            }
                
        }
    }
    std::cout<<"complete get matrix A"<<std::endl;
    std::cout<<"unknowns "<<nUnknowns<<std::endl;
    // Solve the sparse linear system of equations

    Eigen::SparseMatrix<float> A(nUnknowns, nUnknowns);
    A.setFromTriplets(lhsTriplets.begin(), lhsTriplets.end());

    Eigen::SparseLU< Eigen::SparseMatrix<float> > solver;
    solver.analyzePattern(A);
    solver.factorize(A);

    Eigen::MatrixXf result(nUnknowns, channels);

    for (int c = 0; c < channels; ++c)
        result.col(c) = solver.solve(rhs.col(c));
    

    // Copy results back

    for (int y = 0; y < f.rows; ++y) {
        for (int x = 0; x < f.cols; ++x) {
            const cv::Point p(x, y);
            const int pid = unknownIdx(p);

            if (pid > -1) {
                Eigen::Map<Eigen::VectorXf>(r.ptr<float>(p.y, p.x), channels) = result.row(pid);
            }

        }
    }

}

