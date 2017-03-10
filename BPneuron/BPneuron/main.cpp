#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include "unistd.h"

#define Data  5000
#define In 2
#define Out 1
#define Neuron 55
#define TrainC 20000
#define A  0.2
#define B  0.4
#define a  0.2
#define b  0.3

//提高精度再次训练数据
#define A2 0.1
#define B2 0.2
#define a2 0.1
#define b2 0.15

#define A3 0.05
#define B3 0.1
#define a3 0.05
#define b3 0.075

double d_in[Data][In],d_out[Data][Out];                 //数据输入,数据输出
double w[Neuron][In],o[Neuron],v[Out][Neuron];          //网络输入口,隐层网络,输出网络
double Maxin[In],Minin[In],Maxout[Out],Minout[Out];     //最大输入，最小输入，最大输出，最小输出
double OutputData[Out];                                 //输出数据
double dv[Out][Neuron],dw[Neuron][In];                  //输出网络,网络输入
double e;                                               //常量e



//写入测试
void writeTest(){
    FILE *fp1,*fp2;                                     //文件1,文件2
    double r1,r2;                                       //练习数据变量定义
    int i;
    srand((unsigned)time(NULL));                        //设置随机数种子
    if((fp1=fopen("in.txt","w"))==NULL){                //打开in.txt文件
        printf("can not open the in file\n");
        exit(0);
    }
    if((fp2=fopen("out.txt","w"))==NULL){               //打开out.txt文件
        printf("can not open the out file\n");
        exit(0);
    }
    
    for(i=0;i<Data;i++){                                //生成练习数据
        r1=rand()%10000/100.0;                          //生成第一个随机数
        r2=rand()%10000/100.0;                          //生成第二个随机数
        fprintf(fp1,"%lf  %lf\n",r1,r2);                //将两个随机数写入文件
        fprintf(fp2,"%lf \n",r1+r2);                    //将两个随机数相加结果写入文件
    }
    fclose(fp1);                                        //关闭文件
    fclose(fp2);
}


//读取数据
void readData(){
    
    FILE *fp1,*fp2;                                     //文件变量
    int i,j;
    if((fp1=fopen("in.txt","r"))==NULL){                //打开输入文件
        printf("can not open the in file\n");
        exit(0);
    }
    for(i=0;i<Data;i++)                                 //读入数据
        for(j=0; j<In; j++)
            fscanf(fp1,"%lf",&d_in[i][j]);              //将数据写入输入数组
    fclose(fp1);
    
    if((fp2=fopen("out.txt","r"))==NULL){
        printf("can not open the out file\n");
        exit(0);
    }
    for(i=0;i<Data;i++)
        for(j=0; j<Out; j++)
            fscanf(fp1,"%lf",&d_out[i][j]);             //将输出数据写入输出数组
    fclose(fp2);
}

//初始化神经网络
void initBPNework(){
    
    int i,j;
    
    for(i=0; i<In; i++){
        Minin[i]=Maxin[i]=d_in[0][i];
        for(j=0; j<Data; j++)
        {
            Maxin[i]=Maxin[i]>d_in[j][i]?Maxin[i]:d_in[j][i];   //获取输入数据最大值
            Minin[i]=Minin[i]<d_in[j][i]?Minin[i]:d_in[j][i];   //获取输入数据最小值
        }
    }
    
    for(i=0; i<Out; i++){
        Minout[i]=Maxout[i]=d_out[0][i];
        for(j=0; j<Data; j++)
        {
            Maxout[i]=Maxout[i]>d_out[j][i]?Maxout[i]:d_out[j][i];  //获取输出数据最大值
            Minout[i]=Minout[i]<d_out[j][i]?Minout[i]:d_out[j][i];  //获取输出数据最小值
        }
    }
    
    for (i = 0; i < In; i++)
        for(j = 0; j < Data; j++)
            d_in[j][i]=(d_in[j][i]-Minin[i]+1)/(Maxin[i]-Minin[i]+1);       //设置输入数据网络
    
    for (i = 0; i < Out; i++)
        for(j = 0; j < Data; j++)
            d_out[j][i]=(d_out[j][i]-Minout[i]+1)/(Maxout[i]-Minout[i]+1);  //设置输出数据网络
    
    for (i = 0; i < Neuron; ++i)
        for (j = 0; j < In; ++j){
            w[i][j]=rand()*2.0/RAND_MAX-1;                                  //生成输入随机权值
            dw[i][j]=0;                                                     //输入偏差设置为0
        }
    
    for (i = 0; i < Neuron; ++i)
        for (j = 0; j < Out; ++j){
            v[j][i]=rand()*2.0/RAND_MAX-1;                                  //生成输出随机权值
            dv[j][i]=0;                                                     //输出网络偏差设置为0
        }
}

void computO(int var){
    
    int i,j;
    double sum,y;
    for (i = 0; i < Neuron; ++i){
        sum=0;
        for (j = 0; j < In; ++j)
            sum+=w[i][j]*d_in[var][j];
        o[i]=1/(1+exp(-1*sum));                                             //隐层结果计算
    }
    
    for (i = 0; i < Out; ++i){
        sum=0;
        for (j = 0; j < Neuron; ++j)
            sum+=v[i][j]*o[j];
        
        OutputData[i]=sum;                                                  //输出结果计算
    }
}


//线性回归更新节点
void backUpdate(int var)
{
    int i,j;
    double t;
    for (i = 0; i < Neuron; ++i)
    {
        t=0;
        for (j = 0; j < Out; ++j){
            t+=(OutputData[j]-d_out[var][j])*v[j][i];
            
            dv[j][i]=A*dv[j][i]+B*(OutputData[j]-d_out[var][j])*o[i];       //计算偏差值
            v[j][i]-=dv[j][i];                                              //调整输出权值
        }
        
        for (j = 0; j < In; ++j){
            dw[i][j]=a*dw[i][j]+b*t*o[i]*(1-o[i])*d_in[var][j];             //计算输入偏差值
            w[i][j]-=dw[i][j];                                              //调整输入权值
        }
    }
}

//线性回归更新节点
void backUpdate2(int var)
{
    int i,j;
    double t;
    for (i = 0; i < Neuron; ++i)
    {
        t=0;
        for (j = 0; j < Out; ++j){
            t+=(OutputData[j]-d_out[var][j])*v[j][i];
            
            dv[j][i]=A2*dv[j][i]+B2*(OutputData[j]-d_out[var][j])*o[i];       //计算偏差值
            v[j][i]-=dv[j][i];                                              //调整输出权值
        }
        
        for (j = 0; j < In; ++j){
            dw[i][j]=a2*dw[i][j]+b2*t*o[i]*(1-o[i])*d_in[var][j];             //计算输入偏差值
            w[i][j]-=dw[i][j];                                              //调整输入权值
        }
    }
}

//线性回归更新节点
void backUpdate3(int var)
{
    int i,j;
    double t;
    for (i = 0; i < Neuron; ++i)
    {
        t=0;
        for (j = 0; j < Out; ++j){
            t+=(OutputData[j]-d_out[var][j])*v[j][i];
            
            dv[j][i]=A3*dv[j][i]+B3*(OutputData[j]-d_out[var][j])*o[i];       //计算偏差值
            v[j][i]-=dv[j][i];                                              //调整输出权值
        }
        
        for (j = 0; j < In; ++j){
            dw[i][j]=a3*dw[i][j]+b3*t*o[i]*(1-o[i])*d_in[var][j];             //计算输入偏差值
            w[i][j]-=dw[i][j];                                              //调整输入权值
        }
    }
}


//使用网络计算结果
double result(double var1,double var2)
{
    int i,j;
    double sum,y;
    
    var1=(var1-Minin[0]+1)/(Maxin[0]-Minin[0]+1);                           //计算数值分布
    var2=(var2-Minin[1]+1)/(Maxin[1]-Minin[1]+1);
    
    for (i = 0; i < Neuron; ++i){
        sum=0;
        sum=w[i][0]*var1+w[i][1]*var2;                                      //计算隐层值
        o[i]=1/(1+exp(-1*sum));
    }
    sum=0;
    for (j = 0; j < Neuron; ++j)
        sum+=v[0][j]*o[j];                                                  //计算输出值
    
    return sum*(Maxout[0]-Minout[0]+1)+Minout[0]-1;                         //返回计算结果
}

//网络权值记入文件
void writeNeuron()
{
    FILE *fp1;
    int i,j;
    if((fp1=fopen("neuron.txt","w"))==NULL)
    {
        printf("can not open the neuron file\n");
        exit(0);
    }
    for (i = 0; i < Neuron; ++i)
        for (j = 0; j < In; ++j){
            fprintf(fp1,"%lf ",w[i][j]);                                    //输出隐层输入权值到文件
        }
    fprintf(fp1,"\n\n\n\n");
    
    for (i = 0; i < Neuron; ++i)	
        for (j = 0; j < Out; ++j){
            fprintf(fp1,"%lf ",v[j][i]);                                    //输出输出层权值到文件
        }
    
    fclose(fp1);
}


//网络训练
void  trainNetwork(){
    
    int i,c=0,j;
    do{
        e=0;
        for (i = 0; i < Data; ++i){
            computO(i);
            for (j = 0; j < Out; ++j)
                e+=fabs((OutputData[j]-d_out[i][j])/d_out[i][j]);
            backUpdate(i);
        }
        printf("%d  %lf\n",c,e/Data);                                       //输出训练结果和结果误差
        c++;
        usleep(10);
    }while(c<TrainC && e/Data>0.01);
}

//提高精度再训练网络
void  trainNetwork2(){
    
    int i,c=0,j;
    do{
        e=0;
        for (i = 0; i < Data; ++i){
            computO(i);
            for (j = 0; j < Out; ++j)
                e+=fabs((OutputData[j]-d_out[i][j])/d_out[i][j]);
            backUpdate2(i);
        }
        printf("%d  %lf\n",c,e/Data);                                       //输出训练结果和结果误差
        c++;
        usleep(10);
    }while(c<TrainC && e/Data>0.005);
}


//提高精度再训练网络
void  trainNetwork3(){
    
    int i,c=0,j;
    do{
        e=0;
        for (i = 0; i < Data; ++i){
            computO(i);
            for (j = 0; j < Out; ++j)
                e+=fabs((OutputData[j]-d_out[i][j])/d_out[i][j]);
            backUpdate2(i);
        }
        printf("%d  %lf\n",c,e/Data);                                       //输出训练结果和结果误差
        c++;
        usleep(10);
    }while(c<TrainC && e/Data>0.0025);
}


int  main(int argc, char const *argv[])
{
    writeTest();
    readData();
    initBPNework();
    trainNetwork();
    printf("%lf \n",result(56,8) );
    printf("%lf \n",result(2.1,7) );
    printf("%lf \n",result(30.3,8) );
//    trainNetwork2();
//    printf("%lf \n",result(56,8) );
//    printf("%lf \n",result(2.1,7) );
//    printf("%lf \n",result(30.3,8) );
//    trainNetwork3();
//    printf("%lf \n",result(56,8) );
//    printf("%lf \n",result(2.1,7) );
//    printf("%lf \n",result(30.3,8) );
    writeNeuron();
    return 0;
}
