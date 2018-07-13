#!/bin/bash
#脚本以 task1.sh N 的格式运行，根据用户传入的参数N，通过递归函数计算出阶乘N！
s=1     #定义全局变量s

function factorial()       #计算阶乘的递归函数  
{
	n=$1
	if [ $n -ne 1 ]    #当n不为1时，向下递归
	then
		factorial $[$n-1]
	fi
	s=$(($s*$1 ))      #计算阶乘 s=K*（K-1）！，1<=K<=N
}

if [ ! -n "$1" ] ;then     #若无传参，输出usage；有传参则进行计算，并输出结果
    echo -e "usage: task1.sh [n] \ncalculates a number's factorial"
else
   factorial $1
   echo $s 
fi
