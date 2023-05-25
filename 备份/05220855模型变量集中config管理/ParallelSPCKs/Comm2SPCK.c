// COMPILE 
// gcc Comm2SPCK.c -lspck_rt -lrt -lm -L /home/yaoyao/Simpack-2021x/run/realtime/linux64 -L /home/yaoyao/Simpack-2021x/run/bin/linux64 -I /home/yaoyao/Simpack-2021x/run/realtime -o  Comm2SPCK   

// RUN DEMO
// ./Comm2SPCK 9100 12120 0.01 55

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "/home/yaoyao/Simpack-2021x/run/realtime/spck_rt.h"
// TCP includes
#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdbool.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <netinet/tcp.h>

#define DEFAULT_TCP2SPCK_PORT 9999 // 与SPCK的默认通信端口
#define BUFFER_SIZE 1024
#define SPCK_MODE E_RT_MODE__KEEP_SLV 
#define SPCK_RT_PRIO 0 
#define DEFAULT_SPCK_UDP_PORT 12120
#define SPCK_UDP_TIMEOUT 5
#define SPCK_VERBOSE 0
#define DEFAULT_ControlInterval 0.01

int main( int argc, char *argv[] )
{
   const char* name;
   double tend = 999.99; // 单位：秒
   double time;
   double value;
   double wallTime;
   double* u;
   double* y;
   int i;
   int j;
   int ierr;
   int index;
   int nu;
   int ny;
   struct timespec t0;
   struct timespec t1;
   int SPCK_UDP_PORT = DEFAULT_SPCK_UDP_PORT;
   double ControlInterval = DEFAULT_ControlInterval;
   double h = DEFAULT_ControlInterval ; 

   // 根据端口号分配运行的CPU
   char *SPCK_CPUS = malloc(20); 
   // 工作路径
   char Work_path[200];   
   // SPCK路径
   char SPCK_PATH[200] = "/home/yaoyao/Simpack-2021x"; 

   // 解析命令行参数以配置TCP与UDP通信的端口
   int port = DEFAULT_TCP2SPCK_PORT;

   if (argc > 1)
   {
      port = atoi(argv[1]);
   }
   if (argc > 2)
   {
      SPCK_UDP_PORT = atoi(argv[2]);
   }
   if (argc > 3)
   {
      ControlInterval = atof(argv[3]);  // atoi转换为整数,应该使用atof函数
      h = ControlInterval ;
   }
   if (argc > 4)
   {
      tend = atof(argv[4]);
   }
   if (argc > 5)    // 对Work_path的输入
   {
      strncpy(Work_path, argv[5], sizeof(Work_path)-1);
   }
   if (argc > 6)   // 对SPCK_PATH的输入
   {
      strncpy(SPCK_PATH, argv[6], sizeof(SPCK_PATH)-1);
   }

   int NoCPU = port % 100 + 1 ;
   
   sprintf(SPCK_CPUS, "%d", NoCPU);

   char MODEL_FILE[300]; 
   snprintf(MODEL_FILE, sizeof(MODEL_FILE) - 1, "%s/ParallelSPCKs/Model_%d.spck", Work_path, NoCPU);


   /* TCP通信设置 START */
   int sockfd, new_socket;
   struct sockaddr_in serv_addr, cli_addr;
   socklen_t addr_size;
   char buffer[BUFFER_SIZE] = {0};

   // 创建套接字
   sockfd = socket(AF_INET, SOCK_STREAM, 0);
   if (sockfd < 0)
   {
      perror("socket creation failed");
      exit(EXIT_FAILURE);
   }

   // 设置套接字选项，以便在套接字关闭后立即释放端口号
   int opt = 1;
   if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0)
   {
      perror("setsockopt failed");
      exit(EXIT_FAILURE);
   }

   // 设置服务器地址结构
   memset(&serv_addr, 0, sizeof(serv_addr));
   serv_addr.sin_family = AF_INET;
   serv_addr.sin_addr.s_addr = INADDR_ANY;
   serv_addr.sin_port = htons(port);

   // 绑定套接字和地址
   if (bind(sockfd, (const struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
   {
      perror("bind failed");
      exit(EXIT_FAILURE);
   }

   // 监听套接字
   if (listen(sockfd, 1) < 0)
   {
      perror("listen failed");
      exit(EXIT_FAILURE);
   }

   // 接受客户端连接
   addr_size = sizeof(cli_addr);
   new_socket = accept(sockfd, (struct sockaddr *)&cli_addr, &addr_size);
   if (new_socket < 0)
   {
      perror("accept failed");
      exit(EXIT_FAILURE);
   }

   // 设置缓冲区大小
   int buffer_size = 512; // 655360
   setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &buffer_size, sizeof(buffer_size));
   setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &buffer_size, sizeof(buffer_size));
   // 禁用Nagle算法
   int flag = 1;
   setsockopt(new_socket, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int));
   /* TCP通信设置 END */

   ierr = SpckRtInitUDP( SPCK_MODE, SPCK_PATH, MODEL_FILE, SPCK_CPUS, SPCK_RT_PRIO, SPCK_VERBOSE, SPCK_UDP_PORT, SPCK_UDP_TIMEOUT );
   if( ierr )
   {
      SpckRtFinish();
      printf( "SpckRtInitUDP failed\n" );
      return( 1 );
   }
   
   /* Get model dimensions and reserve u/y vector */
   SpckRtGetUYDim( &nu, &ny );
   u = (double*)calloc( nu, sizeof( double ) );
   y = (double*)calloc( ny, sizeof( double ) );

   // Print the dimensions of u and y
   //printf("Dimensions of u: %d\n", nu);
   //printf("Dimensions of y: %d\n", ny);

   if( u == NULL || y == NULL )
   {
      SpckRtFinish();
      printf( "Could not allocate memory for %i inputs and/or %i outputs\n", nu, ny );
      return( 1 );
   }

   /* Start realtime solver */
   if( SpckRtStart() )
   {
      SpckRtFinish();
      printf( "Could not start realtime simulation.\n" );
      return( 1 );
   }

   //printf( "C:开始calculation loop\n" );
   /*calculation loop*/
   time = 0;
   for( i = 0 ; i <= tend/h ; ++i )
   {
      time += h;
      // y变量
      SpckRtGetY( y );
      //printf( "C:Rev Y From SPCKrt\n" );
      uint32_t ny_network = htonl(ny); 
      send(new_socket, &ny_network, sizeof(ny_network), 0);
      for (int k = 0; k < ny; ++k) {
         float y_value = y[k];
         uint32_t y_value_network = htonl(*(uint32_t *)&y_value);
         send(new_socket, &y_value_network, sizeof(y_value_network), 0);
      }
      // u变量
      double u_num[4];
      const double *u = u_num;
      ssize_t recv_len;  
      memset(buffer, 0, BUFFER_SIZE);
      recv_len = recv(new_socket, buffer, BUFFER_SIZE, 0); 

      // 检查recv_len的值，如果为0，则关闭连接并退出循环
      if (recv_len == 0) {  
         //printf( "C:因Actor客户端主动关闭TCP连接，提前停止Simpack仿真\n" );
         SpckRtFinish();
         shutdown(new_socket, SHUT_RDWR); // 可以执行到
         close(new_socket);
         close(sockfd);
         return( 0 );
         // break;

      } else {
         sscanf(buffer, "%lf %lf %lf %lf", &u_num[0], &u_num[1], &u_num[2], &u_num[3]);
      }
      SpckRtSetU(u);
      //printf("C:Sent U to SPCKrt\n");
      if( SpckRtAdvance( time ) )
      {
         SpckRtFinish();
         printf( "SpckRtAdvance failed.\n" );
         return( 1 );
      }

   }

   //关闭Socket
   shutdown(new_socket, SHUT_RDWR);
   close(new_socket);
   close(sockfd); 

   /* Close SIMPACK Realtime */
   SpckRtFinish();

   //printf( "Finished.\n" );
   return( 0 );
}
