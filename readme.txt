本实验由c++代码完成，同时实现了普通的泊松合成和加速后的泊松合成。
实验文档在根目录的doc目录下。

依赖opencv库以及Eigen库，前者用来处理图象，后者用来解方程。
makefile中-I /usr/local/include/eigen3为mac下本机的Eigen库位置，在其他环境编译时请更改为适合的路径。

test1,test2,test3,test4中为测试用例，推荐使用的offsetx，offsety参数保存在para.txt中。

seamless_cloning.cpp中包含main函数，其中调用了加速后的函数和加速前的函数。
如果想要在测试的过程中取消普通的函数，修改main函数即可。