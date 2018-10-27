## TransparentMirror

### 环境配置
注：给定版本号的最好安装该版本或更高版本，没给版本号安装最新版本即可。

* Ubuntu 16.04        // 系统
* CUDA 8.0            // GPU并行计算
* VTK 7.1             // QVTKWidget点云显示
* PCL 1.8.0           // 点云库
* QT 5.9              // GUI界面显示
* freenect2           // 读取Kinect-2点云数据

### 安装说明
* CUDA

	按照官网说明安装，最好选8.0，9.0编译本程序时会有错误。

* QT + VTK + PCL

	注意安装顺序，先安装QT，再VTK，最后PCL
安装过程中，明确指定依赖库的路径，保证依赖库的一致性。例如，编译VTK时，
明确指定QMake路径以及其他相关路径（QTCore_DIR等）为之前安装的Qt的路径；后续
要用到Qt库的话，统一指定为安装的Qt的路径。

	参考链接 [http://unanancyowen.com/en/pcl-with-qt/]()
该链接虽然是在Windows下环境编译的，但是相关选项的设置适合Ubuntu环境。

* freenect2
	该库用于读取Kinect-2的点云数据，其代码托管在Github，按照网页上的说明安装。
	[https://github.com/OpenKinect/libfreenect2]()

### 软件架构说明

#### 工程管理

采用CMake管理工程，工程文件为`CMakeLists.txt`。

在该文件中明确指定各依赖库的路径，否则会采用系统默认的库，从而带来编译错误。

环境配置时安装路径的设置可能不同，按需对路径做出相应的修改。

#### 功能模块

* **cuda文件夹**：运用了GPU并行计算功能的代码，用于特征点的提取；
* **gui文件夹**：用于GUI界面显示的代码；
* **Models文件夹**：测试的模型。运行时，需要将该文件夹复制到可执行文件所在的目录；
* **thread文件夹**：耗时的、多线程处理代码，用于获取点云数据和位姿识别；

* **软件根目录**：

  **distance_measurer**     // 计算一个点到一个点云之间的最短距离
  
  **feature_point_extractor**       // 提取特征点，cuda文件夹有个对应的并行计算版本
  
  **icp**       // Iteractive Closest Point算法
  
  **kinect2_grabber**       // 获取Kinect-2点云数据
  
  **lgf_corresp_group**     // Local Geometric Feature（LGF）特征匹配
  
  **lgf_estimator**         // LGF生成算法
  
  **local_geometric_feature**       // LGF类的定义
  
  **main**      // 主函数
  
  **segmentor**     // 点云分割

